import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import re
import logging
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

class MedTexEngine:
    def __init__(self):
        # Model 1: Med7 for Medication details
        self.med7 = spacy.load("en_core_med7_lg")
        # Model 2: SciSpacy for Anatomy and Chemicals
        self.sci = spacy.load("en_ner_bionlp13cg_md")
        # Model 3: BC5CDR for diseases
        try:
            self.sci_bc5cdr = spacy.load("en_ner_bc5cdr_md")
            logger.info("✅ SciSpaCy BC5CDR loaded")
        except OSError:
            self.sci_bc5cdr = None
            logger.warning("⚠️ en_ner_bc5cdr_md not found")
        
        # Drug vocabulary for validation (300+ drugs across therapeutic classes)
        self.KNOWN_DRUGS = {
            # Analgesics / NSAIDs
            "paracetamol", "acetaminophen", "ibuprofen", "naproxen", "diclofenac", "aceclofenac",
            "aspirin", "celecoxib", "indomethacin", "ketorolac", "mefenamic acid", "piroxicam",
            "meloxicam", "etoricoxib", "nimesulide", "sulindac", "flurbiprofen", "ketoprofen",
            # Antibiotics
            "amoxicillin", "amoxicillin-clavulanate", "co-amoxiclav", "ampicillin", "cloxacillin",
            "azithromycin", "clarithromycin", "erythromycin", "ciprofloxacin", "levofloxacin",
            "moxifloxacin", "ofloxacin", "doxycycline", "tetracycline", "minocycline",
            "metronidazole", "tinidazole", "cephalexin", "cefixime", "cefuroxime", "ceftriaxone",
            "cefpodoxime", "cefepime", "ceftazidime", "nitrofurantoin", "trimethoprim",
            "sulfamethoxazole", "clindamycin", "vancomycin", "linezolid", "gentamicin",
            # Cardiovascular
            "amlodipine", "nifedipine", "diltiazem", "verapamil", "atenolol", "metoprolol",
            "bisoprolol", "carvedilol", "propranolol", "lisinopril", "enalapril", "ramipril",
            "benazepril", "quinapril", "fosinopril", "losartan", "valsartan", "telmisartan",
            "irbesartan", "candesartan", "olmesartan", "hydrochlorothiazide", "furosemide",
            "spironolactone", "eplerenone", "torsemide", "bumetanide", "atorvastatin",
            "rosuvastatin", "simvastatin", "pravastatin", "fluvastatin", "pitavastatin",
            "clopidogrel", "warfarin", "apixaban", "rivaroxaban", "dabigatran", "digoxin",
            "isosorbide", "nitroglycerin", "ranolazine", "ivabradine",
            # Diabetes
            "metformin", "glibenclamide", "glimepiride", "glipizide", "gliclazide", "sitagliptin",
            "vildagliptin", "empagliflozin", "dapagliflozin", "canagliflozin", "insulin",
            "insulin glargine", "insulin lispro", "insulin aspart", "insulin detemir",
            "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "exenatide", "liraglutide",
            # GI
            "omeprazole", "pantoprazole", "esomeprazole", "lansoprazole", "rabeprazole",
            "ranitidine", "famotidine", "cimetidine", "domperidone", "metoclopramide",
            "ondansetron", "granisetron", "palonosetron", "loperamide", "dicyclomine",
            "hyoscine", "scopolamine", "bismuth", "sucralfate", "mesalamine", "balsalazide",
            # Respiratory
            "salbutamol", "albuterol", "salmeterol", "formoterol", "budesonide", "fluticasone",
            "beclomethasone", "montelukast", "zafirlukast", "cetirizine", "loratadine",
            "fexofenadine", "chlorpheniramine", "diphenhydramine", "ipratropium", "tiotropium",
            "theophylline", "dextromethorphan", "guaifenesin", "ambroxol", "acetylcysteine",
            # CNS / Psychiatry
            "sertraline", "fluoxetine", "escitalopram", "paroxetine", "citalopram", "venlafaxine",
            "duloxetine", "amitriptyline", "nortriptyline", "imipramine", "desipramine",
            "alprazolam", "diazepam", "clonazepam", "lorazepam", "oxazepam", "zolpidem",
            "eszopiclone", "haloperidol", "risperidone", "olanzapine", "quetiapine",
            "aripiprazole", "clozapine", "lithium", "valproate", "valproic acid", "carbamazepine",
            "phenytoin", "levetiracetam", "gabapentin", "pregabalin", "topiramate", "lamotrigine",
            "donepezil", "memantine", "rivastigmine", "galantamine", "modafinil", "methylphenidate",
            # Vitamins / Supplements
            "vitamin d", "calcium", "ferrous sulfate", "folic acid", "vitamin b12", "cyanocobalamin",
            "zinc", "multivitamin", "iron", "magnesium", "potassium", "sodium", "omega-3",
            "fish oil", "coenzyme q10", "vitamin c", "ascorbic acid", "vitamin e", "tocopherol",
            # Thyroid
            "levothyroxine", "thyroxine", "liothyronine", "methimazole", "propylthiouracil",
            # Steroids
            "prednisolone", "prednisone", "dexamethasone", "methylprednisolone", "hydrocortisone",
            "betamethasone", "triamcinolone", "fluticasone", "budesonide", "beclomethasone",
            # Topical / Derma
            "clotrimazole", "miconazole", "fluconazole", "terbinafine", "ketoconazole",
            "acyclovir", "valacyclovir", "famciclovir", "mupirocin", "bacitracin", "neomycin",
            "hydroquinone", "tretinoin", "adapalene", "benzoyl peroxide", "salicylic acid",
            # Antihistamines
            "cetirizine", "levocetirizine", "loratadine", "desloratadine", "fexofenadine",
            "chlorpheniramine", "diphenhydramine", "promethazine", "dimenhydrinate",
            # Brand names (common South-Asian market)
            "phexin", "zeedol", "zerodol", "stolin", "plax", "crocin", "combiflam", "dolo",
            "pantodac", "pan", "razo", "nexpro", "nexium", "omeday", "voveran", "brufen",
            "augmentin", "taxim", "ceftas", "zithromax", "ciplox", "cifran", "flagyl", "metrogyl",
            "glucophage", "januvia", "glycomet", "amaryl", "obimet", "piozone",
            "amlip", "stamlo", "amcard", "telma", "olsar", "losar", "covance", "cardace",
            "atorva", "rosuvas", "crestor", "lipitor", "ecosprin", "clopivas", "tromday",
            "sorbitrate", "ismo", "lasix", "aldactone", "dytor", "frumil",
            "stemetil", "perinorm", "emeset", "ondansetron", "enteroquinol", "eldoper",
            "asthalin", "budecort", "seroflo", "foracort", "duolin", "combivent", "montek",
            "allegra", "cetirizine", "okacet", "ctd", "lcz",
            "disprin", "metacin", "flexon", "ibugesic", "brufen", "volini",
            "amphojel", "amphogel", "belladonna", "belledonna",
            "depsonil", "imipramine", "tofranil", "lonazep", "eptoin", "encorate",
            "gabapin", "lyrica", "pregeb", "valparin",
            "thyronorm", "eltroxin", "neomercazole",
            "wysolone", "dexa", "dexona", "medrol", "betnesol",
            "candid", "nizoral", "sporanox", "zovirax",
            # Additional common medications
            "colgate", "amphojel", "amphogel", "belladonna",
        }
        
        # Also include normalized forms for validation
        self.KNOWN_DRUGS.update({
            "cephalexin", "aceclofenac", "clopidogrel", "belladonna", "stolin"
        })
        
        # Drug name normalization map (brand/generic → standard name)
        self.DRUG_NORMALIZATION = {
            "phexin": "cephalexin",
            "zeedol": "aceclofenac",
            "zerodol": "aceclofenac",
            "stolin": "stolin",
            "plax": "clopidogrel",
            "belledonna": "belladonna",
            "amphojel": "amphojel",
            "disprin": "aspirin",
            "crocin": "paracetamol",
            "dolo": "paracetamol",
            "tylenol": "paracetamol",
            "advil": "ibuprofen",
            "motrin": "ibuprofen",
            "brufen": "ibuprofen",
            "glucophage": "metformin",
            "glycomet": "metformin",
            "prilosec": "omeprazole",
            "lipitor": "atorvastatin",
            "atorva": "atorvastatin",
            "zestril": "lisinopril",
            "prinivil": "lisinopril",
            "lopressor": "metoprolol",
            "toprol": "metoprolol",
            "norvasc": "amlodipine",
            "stamlo": "amlodipine",
            "amlip": "amlodipine",
            "lasix": "furosemide",
            "rosuvas": "rosuvastatin",
            "crestor": "rosuvastatin",
            "zithromax": "azithromycin",
            "ciplox": "ciprofloxacin",
            "cifran": "ciprofloxacin",
            "flagyl": "metronidazole",
            "metrogyl": "metronidazole",
            "augmentin": "amoxicillin-clavulanate",
            "asthalin": "salbutamol",
            "montek": "montelukast",
            "thyronorm": "levothyroxine",
            "eltroxin": "levothyroxine",
            "wysolone": "prednisolone",
            "medrol": "methylprednisolone",
            "lyrica": "pregabalin",
            "pregeb": "pregabalin",
            "januvia": "sitagliptin",
            "amaryl": "glimepiride",
        }
        
        # OCR correction dictionary
        self.OCR_CORRECTIONS = {
            "zee dol": "zeedol",
            "zee odol": "zeedol",
            "p hexin": "phexin",
            "sto lin": "stolin",
            "col gate": "colgate",
            "oral b": "oralb",
            "electric brush": "electricbrush",
            "tooth brush": "toothbrush",
            "gate plax": "plax",
            "col plax": "plax",
            "belladonna": "belladonna",
            "amphogel": "amphogel",
            "amphojel": "amphojel",
            # Severe OCR corruption corrections
            "amphoel": "amphojel",
            "belledonna": "belladonna",
            "amphojel qs": "amphojel",
            "amphojel qs ad": "amphojel",
            "amphoel qs": "amphojel",
            "amphoel qs ad": "amphojel",
            # Unit corrections (me → mg, common OCR error)
            " me": " mg",
            "me ": "mg ",
            " me ": " mg ",
        }
        
        # Model 3: Transformer NER (biomedical-ner-all) - secondary model
        self.transformer_ner = None
        if os.getenv("ENABLE_HEAVY_MODELS", "false").lower() == "true":
            try:
                model_name = os.getenv("CLINICAL_NER_MODEL", "d4data/biomedical-ner-all")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                self.transformer_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
                print(f"✅ Transformer NER loaded ({model_name})")
            except Exception as e:
                print(f"⚠️ Transformer NER not loaded: {e}")
        
        # ENTITY_MAP: Standardized clinical entity labels
        self.ENTITY_MAP = {
            # Med7 labels (primary)
            "DRUG": "DRUG",
            "DOSAGE": "DOSAGE",
            "DURATION": "DURATION",
            "FORM": "FORM",
            "FREQUENCY": "FREQUENCY",
            "ROUTE": "ROUTE",
            "STRENGTH": "STRENGTH",
            # SciSpacy labels
            "CHEMICAL": "DRUG",
            "ANATOMY": "ANATOMY",
            "DISEASE": "DISEASE",
            "CANCER": "CANCER",
            # BC5CDR labels
            "Chemical": "DRUG",
            "Disease": "DISEASE",
            # Transformer NER labels
            "Medication": "DRUG",
            "Dose": "DOSAGE",
            "Frequency": "FREQUENCY",
            "Route": "ROUTE",
            "Duration": "DURATION",
            "Condition": "DISEASE",
            "Symptom": "SYMPTOM",
        }
        
        # Learning flywheel: learned corrections from DB
        self.learned_corrections = {}  # original_term -> corrected_term
        self.learned_entities = {}  # term -> label (DRUG/DISEASE/SYMPTOM)
        
        # Shorthand expansion map
        self.SHORTHAND_MAP = {
            r"\bod\b": "once daily",
            r"\bbd\b": "twice daily",
            r"\btds\b": "three times daily",
            r"\btid\b": "three times daily",
            r"\bqid\b": "four times daily",
            r"\bqhs\b": "at bedtime",
            r"\bhs\b": "at bedtime",
            r"\bprn\b": "as needed",
            r"\bsos\b": "as needed",
            r"\bstat\b": "immediately",
            r"\bac\b": "before meals",
            r"\bpc\b": "after meals",
            r"\bpo\b": "by mouth",
            r"\biv\b": "intravenous",
            r"\bim\b": "intramuscular",
            r"\bsc\b": "subcutaneous",
            r"\bsl\b": "sublingual",
            r"\bpr\b": "per rectum",
            r"\btop\b": "topically",
            r"\b1-0-1\b": "twice daily",
            r"\b1-1-1\b": "three times daily",
            r"\b1-0-0\b": "once daily morning",
            r"\b0-0-1\b": "once daily night",
            r"\b0-1-0\b": "once daily afternoon",
            r"\b1-1-0\b": "twice daily morning and afternoon",
        }
        
        # Compile shorthand patterns once
        self._compiled_shorthand = [
            (re.compile(pat, re.IGNORECASE), replacement)
            for pat, replacement in self.SHORTHAND_MAP.items()
        ]
        
        # Color map for frontend highlighting
        self.color_map = {
            "DRUG": "#ffcfcc", "DOSAGE": "#d3f9d8", "STRENGTH": "#b2f2bb",
            "DURATION": "#fff3bf", "FORM": "#eebefa", "FREQUENCY": "#d0ebff",
            "ROUTE": "#fff4e6", "DISEASE": "#ffc9c9", "SYMPTOM": "#ffa8a8",
            "ANATOMY": "#c5f6fa", "PROCEDURE": "#ffe8cc", "LAB": "#e3fafc",
            "CHEMICAL": "#ffec99", "CANCER": "#ff6b6b",
        }
        
        logger.info("MedTexEngine initialized")
    
    def load_knowledge_from_db(self, db) -> None:
        """
        Pull all verified corrections from MedicalKnowledge table and merge
        into the in-memory vocabulary. Call this on startup and after /verify.
        """
        from backend.app.core.database import MedicalKnowledge
        try:
            records = db.query(MedicalKnowledge).filter(
                MedicalKnowledge.confidence >= 1
            ).all()
            
            new_corrections = {}
            new_drugs = set()
            new_entities = {}
            
            for rec in records:
                orig = rec.original_term.lower().strip()
                corrected = rec.corrected_term.strip()
                category = (rec.category or "").lower()
                
                if orig != corrected.lower():
                    new_corrections[orig] = corrected
                
                if category in ("drug", "medication"):
                    new_drugs.add(orig)
                    new_drugs.add(corrected.lower())
                    if orig != corrected.lower():
                        new_corrections[orig] = corrected
                
                if category == "entity":
                    # stored as "term::LABEL" in corrected_term
                    if "::" in corrected:
                        term, label = corrected.split("::", 1)
                        new_entities[orig] = label.upper()
            
            # Update in-memory structures
            self.learned_corrections.update(new_corrections)
            self.learned_entities.update(new_entities)
            self.KNOWN_DRUGS.update(new_drugs)
            self.DRUG_NORMALIZATION.update(new_corrections)
            
            logger.info(
                f"Knowledge loaded: {len(new_corrections)} corrections, "
                f"{len(new_drugs)} drug terms, {len(new_entities)} entity labels"
            )
        except Exception as exc:
            logger.warning(f"load_knowledge_from_db failed: {exc}")
    
    def _preprocess(self, text: str) -> str:
        """Clean and normalise raw OCR / clinical text before NER."""
        
        # 1. Apply learned corrections from DB (highest priority)
        for wrong, right in self.learned_corrections.items():
            text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
        
        # 2. Expand medical shorthand (BD → twice daily, etc.)
        for compiled_pat, replacement in self._compiled_shorthand:
            text = compiled_pat.sub(replacement, text)
        
        # 3. Fix merged units (500mg → 500 mg)
        text = re.sub(r"(\d+)(mg|ml|g|mcg|iu|units?)\b", r"\1 \2", text, flags=re.IGNORECASE)
        
        # 4. Fix CamelCase merges
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        
        # 5. Remove ## sub-word artifacts
        text = re.sub(r"##", "", text)
        
        # 6. Normalise whitespace
        text = re.sub(r"[ \t]+", " ", text).strip()
        
        return text
    
    def _format_ent(self, ent, label=None, score=1.0):
        """Format an entity for output."""
        lbl = label or ent.label_
        mapped_lbl = self.ENTITY_MAP.get(lbl, lbl)
        return {
            "text": ent.text,
            "label": mapped_lbl,
            "start": ent.start_char,
            "end": ent.end_char,
            "score": score,
            "color": self.color_map.get(mapped_lbl, "#f0f0f0")
        }
    
    def _extract_medical_patterns(self, text: str) -> list:
        """Rule-based extraction for common clinical patterns."""
        patterns = []
        text_lower = text.lower()

        # Rule-based drug name extraction for known drugs that NER might miss
        for drug in self.KNOWN_DRUGS:
            # Case-insensitive search for drug names (more permissive matching)
            # Try word boundary first, then fallback to substring match for OCR artifacts
            drug_regex = r'\b' + re.escape(drug) + r'\b'
            matches = list(re.finditer(drug_regex, text_lower))
            
            # If no word boundary match, try substring match for OCR artifacts
            # This handles cases like "Amphojel qsad" where OCR adds noise
            if not matches:
                drug_regex = re.escape(drug)
                matches = list(re.finditer(drug_regex, text_lower))
            
            for match in matches:
                # Extract the actual matched text from original text (preserves case)
                matched_text = text[match.start():match.end()]
                # Only add if we haven't already matched this exact position
                # (prevents duplicates from overlapping matches)
                if not any(p["start"] == match.start() and p["end"] == match.end() for p in patterns):
                    patterns.append({
                        "text": matched_text,
                        "label": "DRUG",
                        "start": match.start(),
                        "end": match.end(),
                        "score": 0.9,  # High confidence for rule-based matches
                        "color": self.color_map.get("DRUG", "#ffcfcc")
                    })

        # Fuzzy matching for severely corrupted drug names
        # Split text into words and check each against known drugs
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        for word in words:
            # Skip if already matched
            if any(p["start"] <= text_lower.find(word) and p["end"] >= text_lower.find(word) + len(word) for p in patterns):
                continue
            
            # Use fuzzy matching to find similar drug names
            match = process.extractOne(word, self.KNOWN_DRUGS, scorer=fuzz.ratio)
            if match and match[1] >= 75:  # 75% similarity threshold
                drug_name = match[0]
                # Find the position in original text
                pos = text_lower.find(word)
                if pos >= 0:
                    patterns.append({
                        "text": text[pos:pos+len(word)],
                        "label": "DRUG",
                        "start": pos,
                        "end": pos + len(word),
                        "score": 0.8,  # Slightly lower confidence for fuzzy matches
                        "color": self.color_map.get("DRUG", "#ffcfcc")
                    })

        # Dosage (mg, ml, g, etc.)
        dosage_regex = r'\b\d+\s?(mg|ml|g|mcg|unit|units|tab|tabs|cap|caps)\b'
        for match in re.finditer(dosage_regex, text_lower):
            patterns.append({
                "text": text[match.start():match.end()],
                "label": "DOSAGE",
                "start": match.start(),
                "end": match.end(),
                "score": 1.0,
                "color": self.color_map.get("DOSAGE", "#d3f9d8")
            })

        # Frequency (daily, BD, TDS, etc.)
        freq_regex = r'(once daily|twice daily|thrice daily|every \d+ hours|od|bd|tds|qid|qhs|prn)'
        for match in re.finditer(freq_regex, text_lower):
            patterns.append({
                "text": text[match.start():match.end()],
                "label": "FREQUENCY",
                "start": match.start(),
                "end": match.end(),
                "score": 1.0,
                "color": self.color_map.get("FREQUENCY", "#d0ebff")
            })

        # Dosage Frequency Patterns (e.g., 1-0-1, 1 - 1 - 0)
        freq_pattern_regex = r'\b\d\s*-\s*\d\s*-\s*\d\b'
        for match in re.finditer(freq_pattern_regex, text_lower):
            patterns.append({
                "text": text[match.start():match.end()],
                "label": "FREQUENCY",
                "start": match.start(),
                "end": match.end(),
                "score": 1.0,
                "color": self.color_map.get("FREQUENCY", "#d0ebff")
            })

        return patterns

    def _clean_entities(self, entities: list) -> list:
        """
        Correction layer to fix common model misclassifications.
        Order matters: Strength must be checked BEFORE frequency.
        """
        cleaned = []
        i = 0
        while i < len(entities):
            e = entities[i]
            text = e["text"].lower()
            label = e["label"].lower()

            # Fix label mapping: "Dosage Form" -> "FORM"
            if label == "dosage form":
                e["label"] = "FORM"
                e["color"] = self.color_map.get("FORM", "#eebefa")
                label = "form"

            # FIX: Distinguish STRENGTH vs DOSAGE
            # STRENGTH = concentration (500 mg, 250 mg/5 ml, 3%, 125 mg/5 ml)
            # DOSAGE = quantity to take (1 tablet, 2 caps, 1 tsp, 20 drops)
            freq_words = ['daily', 'twice', 'once', 'thrice', 'nightly', 'weekly', 'monthly', 'hourly']
            
            # Check for STRENGTH pattern (concentration with unit)
            if re.match(r'\d+\s*(mg|ml|g|mcg|µg|iu|mmol|meq|units?)\b', text) and not any(fw in text for fw in freq_words):
                # Additional check: if it starts with a number and has a unit, it's likely strength
                # But if it has dosage form words (tablet, cap, etc.), it's DOSAGE
                dosage_forms = ['tablet', 'tab', 'capsule', 'cap', 'sachet', 'puff', 'drop', 'suppository', 'patch', 'syrup', 'suspension']
                if not any(df in text for df in dosage_forms):
                    e["label"] = "STRENGTH"
                    e["color"] = self.color_map.get("STRENGTH", "#b2f2bb")
            
            # Check for DOSAGE pattern (quantity + form)
            elif re.match(r'\d+\s*(tablet|tab|capsule|cap|sachet|puff|drop|suppository|patch)s?\b', text, re.IGNORECASE):
                e["label"] = "DOSAGE"
                e["color"] = self.color_map.get("DOSAGE", "#d3f9d8")
            
            # Check for DOSAGE pattern (quantity + volume measure)
            elif re.match(r'\d+\s*(tsp|tbsp|ml|teaspoon|tablespoon)\b', text, re.IGNORECASE):
                e["label"] = "DOSAGE"
                e["color"] = self.color_map.get("DOSAGE", "#d3f9d8")

            # Fix merged tokens like "500freq" → split and correct label
            elif re.match(r'\d+[a-z]+', text):
                parts = re.findall(r'(\d+|[a-z]+)', text, re.I)
                if len(parts) == 2:
                    num_part, word_part = parts
                    if word_part.lower() in ['freq', 'frequency']:
                        e["text"] = num_part
                        e["label"] = "DOSAGE"
                        e["color"] = self.color_map.get("DOSAGE", "#ffc9c9")

            # Fix merged patterns like "1 - 0 - 1 x 5" → split into frequency and duration
            elif re.match(r'\d\s*-\s*\d\s*-\s*\d\s*x\s*\d+', text):
                # Extract frequency part
                freq_match = re.search(r'\d\s*-\s*\d\s*-\s*\d', text)
                if freq_match:
                    e["text"] = freq_match.group().replace(' ', '')
                    e["label"] = "FREQUENCY"
                    e["color"] = self.color_map.get("FREQUENCY", "#d0ebff")

            # Fix duration (e.g., "3days", "for 5 days") - MUST have a number
            elif re.match(r'\d+\s*(days?|weeks?|months?)', text):
                e["label"] = "DURATION"
                e["color"] = self.color_map.get("DURATION", "#fff3bf")

            # Fix frequency patterns (e.g., "1-0-1", "1 - 1 - 0")
            elif re.match(r'\d-\d-\d', text):
                e["label"] = "FREQUENCY"
                e["color"] = self.color_map.get("FREQUENCY", "#d0ebff")

            # Fix ROUTE misclassification: if labeled ROUTE but contains known drug, it's DRUG
            if label == "route":
                # Check if any word in the text is a known drug
                words = text.split()
                for word in words:
                    if word in self.KNOWN_DRUGS or word in self.DRUG_NORMALIZATION:
                        e["label"] = "DRUG"
                        e["color"] = self.color_map.get("DRUG", "#ffcfcc")
                        break
                # If still ROUTE, only keep if it's a valid route word
                if e["label"] == "ROUTE" and text not in ["oral", "iv", "topical", "mouth", "eye", "ear"]:
                    e["label"] = "DRUG"  # Default to DRUG for unknown routes that might be drug names
                    e["color"] = self.color_map.get("DRUG", "#ffcfcc")
            
            # Fix route (explicit route words)
            elif text in ["oral", "iv", "topical", "mouth", "eye", "ear"]:
                e["label"] = "ROUTE"
                e["color"] = self.color_map.get("ROUTE", "#fff4e6")
            
            # Fix form (including common OCR fragments)
            elif text in ["tablet", "tab", "capsule", "cap", "syrup", "syr", "injection", "inj", "ointment", "cream", "amp", "ampoule", "sol", "solution"]:
                e["label"] = "FORM"
                e["color"] = self.color_map.get("FORM", "#eebefa")

            # FIX: Split entities that contain FORM + DRUG (e.g., "Solution Amphogel")
            # Check if entity text contains a known FORM word followed by a known DRUG name
            if e["label"] == "DRUG":
                form_words = ["solution", "tablet", "tab", "capsule", "cap", "syrup", "injection", "inj", "ointment", "cream", "amp", "ampoule"]
                for form_word in form_words:
                    if text.startswith(form_word + " "):
                        # Split into FORM and DRUG
                        drug_part = text[len(form_word):].strip()
                        # Check if drug part is a known drug
                        if drug_part in self.KNOWN_DRUGS or drug_part in self.DRUG_NORMALIZATION:
                            # Create FORM entity
                            form_entity = {
                                "text": e["text"][:len(form_word)],
                                "label": "FORM",
                                "start": e["start"],
                                "end": e["start"] + len(form_word),
                                "score": e.get("score", 1.0),
                                "color": self.color_map.get("FORM", "#eebefa")
                            }
                            # Update current entity to just the drug part
                            e["text"] = e["text"][len(form_word):].strip()
                            e["start"] = e["start"] + len(form_word) + 1
                            cleaned.append(form_entity)
                        break

            cleaned.append(e)
            i += 1
        return cleaned

    def _filter_noise(self, entities: list) -> list:
        """
        Filter out garbage entities typically caused by OCR noise and non-medical keywords.
        Enhanced with confidence scoring and strict drug validation.
        """
        NON_MEDICAL = {
            "gum", "paint", "brush", "electric brush", "wash", "electric", "pt",
            "col", "gate", "b pro", "m ft", "ft", "qs", "qsad",
            "toothbrush", "colgate", "oralb", "dental",
        }

        # These are real medical abbreviations — never filter them
        KEEP_SHORT = {"bd", "od", "tds", "qid", "prn", "sos", "sr", "er", "xl", "xr"}

        # Words that are obviously not clinical entities regardless of model label
        NON_CLINICAL_KEYWORDS = {
            "brush", "electric", "toothbrush", "oral b", "colgate",
            "pro 2", "2000n", "tooth", "dental",
        }
        
        # Known garbage patterns from OCR (random character combinations without vowels)
        GARBAGE_PATTERNS = re.compile(r'^[bcdfghjklmnpqrstvwxyz]{2,}$', re.I)
        
        # Common OCR noise fragments that should be filtered
        OCR_NOISE_FRAGMENTS = {"qsad", "ft", "m ft", "qs", "ad"}
        
        filtered = []
        for e in entities:
            text = e["text"].strip()
            text_lower = text.lower()
            
            # Filter subword artifacts (## tokens from transformer models)
            if "##" in text:
                continue
            
            # Filter very short fragments (less than 3 chars) unless known abbreviation
            if len(text) < 3 and text_lower not in KEEP_SHORT:
                continue
            
            # Filter pure digits
            if text.isdigit():
                continue
            
            # Filter non-medical keywords
            if text_lower in NON_MEDICAL:
                continue
            
            # Filter common OCR noise fragments
            if text_lower in OCR_NOISE_FRAGMENTS:
                continue
            
            # Filter CANCER / DISEASE false positives for clearly non-clinical text
            if e["label"] in ("CANCER", "DISEASE", "ANATOMY"):
                if any(kw in text_lower for kw in NON_CLINICAL_KEYWORDS):
                    continue
                # "Oral B Pro 2 2000N electric brush" style — reject if > 4 words and no drug
                words = text.split()
                if len(words) >= 4 and not any(w.lower() in self.KNOWN_DRUGS for w in words):
                    continue

            # Filter garbage patterns (no vowels = likely OCR noise like "qsad", "m ft")
            # Exception: keep if it's a known drug or medical abbreviation
            if GARBAGE_PATTERNS.match(text) and text_lower not in self.KNOWN_DRUGS and text_lower not in KEEP_SHORT:
                continue
            
            # Filter invalid drug names (CRITICAL - uses drug dictionary)
            # Skip this check for rule-based patterns (high confidence 0.9)
            if e["label"] == "DRUG" and e.get("score", 1.0) < 0.9 and not self._is_valid_drug(text):
                continue
            
            # Filter common OCR noise patterns (single letters repeated, etc.)
            if re.match(r'^[a-z]$', text_lower):
                continue
            
            # Confidence scoring filter (VERY IMPORTANT)
            # Reject low-confidence entities unless they're from rule-based patterns
            if e.get("score", 1.0) < 0.6 and e.get("score", 1.0) > 0:
                continue
            
            filtered.append(e)
        
        return filtered

    def _parse_medication_line(self, text: str) -> dict:
        """Extract structured components from a text fragment using precise rules."""
        # Normalize text fragment for internal parsing
        t = text.lower().strip()
        
        # 1. Handle OCR 'fre' or 'mg' misreads for strength
        strength_match = re.search(r'\b(\d+)\s*(mg|ml|g|mcg|fre|q|units?|tab|cap)\b', t, re.I)
        strength = None
        if strength_match:
            val, unit = strength_match.groups()
            # Standardize misread units
            if unit.lower() in ['fre', 'q']: unit = 'mg'
            strength = f"{val} {unit.lower()}"
        else:
            # Fallback: capture number + any word as potential strength
            strength_fallback = re.search(r'\b(\d+)\s+([a-z]+)\b', t, re.I)
            if strength_fallback:
                val, unit = strength_fallback.groups()
                # Only treat as strength if unit looks like a dosage unit
                if unit in ['solution', 'sol', 'ml', 'mg', 'g', 'mcg']:
                    strength = f"{val} {unit}"

        # 2. Precise Frequency (1-0-1)
        frequency = re.search(r'\b\d-\d-\d\b', t)
        
        # 3. Extract frequency patterns like "threetimes daily", "twice daily", "once daily"
        if not frequency:
            freq_patterns = [
                r'\b(three|two|one)\s+times?\s+(daily|day)\b',
                r'\b(thrice|twice|once)\s+daily\b',
                r'\b(three|two|one)\s+times?\s+a\s+day\b',
                r'\b(three|two|one)\s+times?\b'  # Standalone "threetimes"
            ]
            for pattern in freq_patterns:
                freq_match = re.search(pattern, t, re.I)
                if freq_match:
                    frequency = freq_match.group()
                    break

        # 4. Precise Duration (Fix artifacts like 'dadays' or 'x5days' or 'seg 5')
        duration_match = re.search(r'(\d+)\s*(days?|weeks?|months?|seg)\b', t, re.I)
        duration = None
        if duration_match:
            val, unit = duration_match.groups()
            # Normalize 'seg' to 'days' (common OCR artifact)
            if unit.lower() == 'seg': unit = 'days'
            duration = f"{val} {unit.lower()}"
        else:
            # Fallback to unit only if no number found
            unit_only = re.search(r'\b(days?|weeks?|months?)\b', t, re.I)
            if unit_only:
                duration = unit_only.group().lower()

        # 5. Route
        route = re.search(r'\b(oral|iv|topical|eye|ear|mouth)\b', t, re.I)
        
        return {
            "strength": strength,
            "frequency": frequency if isinstance(frequency, str) else (frequency.group() if frequency else None),
            "duration": duration,
            "route": route.group() if route else None
        }

    def _split_mixed_entities(self, entities: list) -> list:
        """
        Split entities that contain multiple components (e.g., "seg 5 threetimes" contains duration + frequency).
        This handles cases where OCR produces a single entity for mixed prescription information.
        """
        split_entities = []
        
        for e in entities:
            text = e["text"].lower()
            label = e["label"]
            
            # Check if this entity might contain multiple components
            # Pattern: duration + frequency (e.g., "seg 5 threetimes", "5 days twice daily")
            has_duration = bool(re.search(r'(\d+)\s*(days?|weeks?|months?|seg)\b', text, re.I))
            has_frequency = bool(re.search(r'(three|two|one)\s+times?|thrice|twice|once|daily', text, re.I))
            
            if has_duration and has_frequency:
                # Split into separate entities
                # Extract duration
                duration_match = re.search(r'(\d+)\s*(days?|weeks?|months?|seg)\b', text, re.I)
                if duration_match:
                    val, unit = duration_match.groups()
                    if unit.lower() == 'seg': unit = 'days'
                    duration_text = f"{val} {unit.lower()}"
                    split_entities.append({
                        "text": duration_text,
                        "label": "DURATION",
                        "start": e["start"],
                        "end": e["start"] + len(duration_text),
                        "score": e.get("score", 1.0),
                        "color": self.color_map.get("DURATION", "#fff3bf")
                    })
                
                # Extract frequency
                freq_match = re.search(r'(three|two|one)\s+times?|thrice|twice|once|daily', text, re.I)
                if freq_match:
                    freq_text = freq_match.group()
                    split_entities.append({
                        "text": freq_text,
                        "label": "FREQUENCY",
                        "start": e["start"] + len(text) - len(freq_text),  # Approximate position
                        "end": e["end"],
                        "score": e.get("score", 1.0),
                        "color": self.color_map.get("FREQUENCY", "#d0ebff")
                    })
            else:
                # Keep entity as is
                split_entities.append(e)
        
        return split_entities

    def _fix_broken_words(self, text: str) -> str:
        """
        Reconstruct words split by OCR or transformer tokenization.
        Simpler, more stable approach using specific drug name patterns.
        """
        # Fix transformer '##' artifacts
        text = re.sub(r'##', '', text)

        # Specific drug name reconstructions (more stable than generic merging)
        text = re.sub(r'\b(zee)\s*(dol)\b', r'zeedol', text, flags=re.I)
        text = re.sub(r'\b(sto)\s*(lin)\b', r'stolin', text, flags=re.I)
        text = re.sub(r'\b(p)\s*(hexin)\b', r'phexin', text, flags=re.I)
        text = re.sub(r'\b(amp)\s*(ho)\s*(jel)\b', r'amphojel', text, flags=re.I)
        text = re.sub(r'\b(belle)\s*(donna)\b', r'belledonna', text, flags=re.I)

        # Fix repeated noise tokens
        text = re.sub(r'(days|weeks|months)\s*(days|weeks|months)', r'\1', text, flags=re.I)

        return text

    def _merge_adjacent_fragments(self, entities: list, text: str) -> list:
        """
        Simplified fragment merging as a safety net for cases where NER still produces
        fragmented entities even after text reconstruction. Less complex than before.
        """
        if not entities:
            return entities
        
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        merged = []
        i = 0
        
        while i < len(sorted_entities):
            current = sorted_entities[i]
            
            # Only merge DRUG entities
            if current['label'] != 'DRUG':
                merged.append(current)
                i += 1
                continue
            
            # Fix merged tokens like "500freq" → split into "500" and "freq"
            if re.search(r'\d+[a-z]+', current["text"], re.I):
                # Split number from word
                parts = re.findall(r'(\d+|[a-z]+)', current["text"], re.I)
                if len(parts) == 2:
                    num_part, word_part = parts
                    if word_part.lower() in ['freq', 'frequency']:
                        # Keep only the number as dosage
                        current["text"] = num_part
                        current["label"] = "DOSAGE"
                    elif word_part.lower() in ['days', 'day', 'week', 'weeks']:
                        # Keep only the number for duration extraction later
                        current["text"] = num_part
                        current["label"] = "DURATION"
            
            # Fix tokens like "2 2000n" → extract the valid dosage part
            if re.search(r'\d+\s+\d+[a-z]', current["text"], re.I):
                # Extract the first number as count, second as dosage
                match = re.search(r'(\d+)\s+(\d+)([a-z]+)', current["text"], re.I)
                if match:
                    count, dosage, unit = match.groups()
                    if unit.lower() in ['mg', 'ml', 'g', 'mcg']:
                        current["text"] = f"{dosage}{unit}"
                        current["label"] = "DOSAGE"
            
            # Look ahead for adjacent DRUG fragments (within 10 chars)
            fragments = [current]
            j = i + 1
            
            while j < len(sorted_entities):
                next_ent = sorted_entities[j]
                
                if next_ent['label'] != 'DRUG':
                    break
                
                gap = next_ent['start'] - current['end']
                if gap <= 10:
                    fragments.append(next_ent)
                    current = next_ent
                    j += 1
                else:
                    break
            
            # If we have multiple fragments, merge them
            if len(fragments) > 1:
                merged_text = ' '.join(f['text'] for f in fragments)
                merged_start = fragments[0]['start']
                merged_end = fragments[-1]['end']
                
                merged_entity = {
                    'text': merged_text,
                    'label': 'DRUG',
                    'start': merged_start,
                    'end': merged_end,
                    'score': max(f.get('score', 0.5) for f in fragments),
                    'color': self.color_map.get('DRUG', '#ffcfcc')
                }
                merged.append(merged_entity)
                i = j
            else:
                merged.append(current)
                i += 1
        
        return merged

    def _clean_drug_name(self, name: str) -> str:
        """Clean and standardize a drug name (capitalize, remove noise, normalize, deduplicate)."""
        if not name: return ""
        # Remove all non-alphabetic characters (keep spaces and hyphens)
        clean = re.sub(r'[^a-zA-Z\s\-]', '', name).strip()
        
        # Remove common noise prefixes: form words AND single-letter OCR artifacts
        strip_prefixes = {"tn", "rx", "tab", "tablet", "cap", "capsule", "inj", "injection",
                          "syrup", "syr", "sol", "solution", "pt", "sr", "xr"}
        words = clean.split()
        while words and words[0].lower() in strip_prefixes:
            words = words[1:]

        # Remove trailing single-char noise (e.g., "Zeedol Pt" → "Zeedol")
        while words and len(words[-1]) <= 2 and not words[-1].lower() in {"er", "sr", "xl", "xr"}:
            words = words[:-1]

        clean = ' '.join(words)
        # Remove duplicate adjacent words (e.g., "phexin phexin" → "phexin")
        seen_words: list = []
        for word in clean.split():
            if not seen_words or word.lower() != seen_words[-1].lower():
                seen_words.append(word)
        clean = ' '.join(seen_words)

        # Normalize to standard drug name if in mapping
        clean_lower = clean.lower()
        if clean_lower in self.DRUG_NORMALIZATION:
            clean = self.DRUG_NORMALIZATION[clean_lower]
        # Also check individual words
        elif len(seen_words) > 1:
            first_word = seen_words[0].lower()
            if first_word in self.DRUG_NORMALIZATION:
                # Use normalized first word + remaining words
                rest = ' '.join(seen_words[1:])
                clean = self.DRUG_NORMALIZATION[first_word] + (' ' + rest if rest else '')
        
        return clean.title() if clean else name.title()

    def _normalize_entities(self, entities: list) -> list:
        """Normalize entity text (e.g., deduplicate drug names in entities)."""
        normalized = []
        for e in entities:
            if e["label"] == "DRUG":
                # First try fuzzy matching if the drug name is corrupted
                cleaned = self._clean_drug_name(e["text"])
                # If cleaned name is not in known drugs, try fuzzy matching
                if cleaned.lower() not in self.KNOWN_DRUGS:
                    match = process.extractOne(cleaned.lower(), self.KNOWN_DRUGS, scorer=fuzz.ratio)
                    if match and match[1] >= 75:
                        # Use the matched drug name
                        e["text"] = match[0].title()
                        e["fuzzy_matched"] = True
                    else:
                        e["text"] = cleaned
                else:
                    e["text"] = cleaned
            normalized.append(e)
        return normalized

    def _advanced_ocr_cleanup(self, text: str) -> str:
        """
        Advanced OCR cleanup with corrections dictionary and pattern fixes.
        This runs BEFORE NER to prevent garbage entities from being created.
        NOTE: We do NOT lowercase here — Med7/SciSpaCy need original casing.
        """
        # Apply corrections dictionary (case-insensitive replace, preserve original case of replacement)
        for wrong, correct in self.OCR_CORRECTIONS.items():
            if wrong not in ["belladonna", "amphogel", "amphojel"]:
                text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
        
        # Fix specific drug name patterns
        text = re.sub(r'zee\s*dol', 'zeedol', text)
        text = re.sub(r'p\s*hexin', 'phexin', text)
        text = re.sub(r'sto\s*lin', 'stolin', text)
        
        # Fix merged dosage patterns (e.g., "500freq" → "500 freq")
        text = re.sub(r'(\d+)(freq)', r'\1 \2', text)
        
        # Fix merged duration patterns (e.g., "5days" → "5 days")
        text = re.sub(r'(\d+)(days?|weeks?|months?)', r'\1 \2', text)
        
        # Fix frequency patterns merged with duration (e.g., "1-0days" → "1-0 days")
        text = re.sub(r'(\d-\d)(days?|weeks?|months?)', r'\1 \2', text)
        
        # Remove non-medical phrases entirely
        non_medical_phrases = ["gum paint", "electric brush", "tooth brush", "col gate", "oral b"]
        for phrase in non_medical_phrases:
            text = text.replace(phrase, "")
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _classify_document(self, text: str) -> str:
        """
        Classify document type based on text patterns.
        Returns: 'prescription', 'clinical_note', 'report', or 'unknown'
        """
        text_lower = text.lower()
        
        # Prescription indicators
        prescription_keywords = ["rx", "tab", "cap", "mg", "dose", "take", "sig", "dispense", "refill"]
        if any(kw in text_lower for kw in prescription_keywords):
            return "prescription"
        
        # Clinical note indicators
        clinical_keywords = ["chief complaint", "history of present illness", "review of systems", "physical exam", "assessment", "plan"]
        if any(kw in text_lower for kw in clinical_keywords):
            return "clinical_note"
        
        # Report indicators
        report_keywords = ["medical report", "laboratory report", "radiology report", "pathology report", "diagnosis"]
        if any(kw in text_lower for kw in report_keywords):
            return "report"
        
        return "unknown"

    def _extract_dosage_patterns(self, text: str) -> list:
        """
        Extract dosage patterns using regex.
        Returns list of dosage strings found.
        """
        # Pattern: number + unit (mg, ml, g, mcg, etc.)
        dosage_pattern = r'\b\d+\s*(mg|ml|g|mcg|µg|units?|tab|cap|tablet|capsule)\b'
        dosages = re.findall(dosage_pattern, text, re.IGNORECASE)
        return dosages

    def _extract_frequency_patterns(self, text: str) -> list:
        """
        Extract frequency patterns using regex.
        Returns list of frequency strings found.
        """
        # Pattern: 1-0-1, 1-1-1, etc.
        numeric_freq = re.findall(r'\b\d-\d-\d\b', text)
        
        # Pattern: once daily, twice daily, etc.
        text_freq = re.findall(r'\b(once|twice|thrice|three times|four times)\s*(daily|weekly|monthly)\b', text, re.IGNORECASE)
        
        return numeric_freq + [f"{a} {b}" for a, b in text_freq]

    def _is_valid_drug(self, name: str) -> bool:
        """
        Validate drug names. Accepts:
         - Names that exactly match (or contain) a known drug
         - Multi-word names where any word is a known drug
         - Names >= 8 chars (long enough to be a real drug name)
         - Names that fuzzy-match a known drug at >= 72%
        Rejects:
         - Pure noise (brush, electric, paint, gum, wash…)
         - Very short names with no vowels
        """
        if not name or len(name) < 3:
            return False

        name_lower = name.lower().strip()

        # Whitelist: exact match
        if name_lower in self.KNOWN_DRUGS:
            return True

        # Check normalization map
        if name_lower in self.DRUG_NORMALIZATION:
            return True

        # Strict noise blacklist (exclude known drugs that are also dental products)
        noise = {
            "brush", "electric", "paint", "wash", "gate", "pro",
            "oralb", "electricbrush", "dental", "toothbrush",
            "col", "hexin", "lin"
        }
        # Note: colgate is in KNOWN_DRUGS and should pass exact match check before this
        if name_lower in noise and name_lower not in self.KNOWN_DRUGS:
            return False

        # Multi-word: accept if ANY word is a known drug
        words = name_lower.split()
        for word in words:
            if word in self.KNOWN_DRUGS or word in self.DRUG_NORMALIZATION:
                return True

        # Accept long-enough names (likely a real multi-word drug/brand)
        if len(name) >= 6:
            return True

        # Short names: require at least one vowel
        if not any(c in "aeiou" for c in name_lower):
            return False

        return False

    def _group_medications(self, entities: list, text: str) -> list:
        """
        Group medication entities using strict boundary segmentation.
        Each drug 'owns' everything until the next drug appears.
        """
        meds = []
        entities = sorted(entities, key=lambda x: x["start"])
        
        # Identify indices of all DRUG entities
        drug_indices = [i for i, e in enumerate(entities) if e["label"] == "DRUG"]

        for idx, drug_i in enumerate(drug_indices):
            drug_ent = entities[drug_i]
            
            # Clean and validate the drug name
            cleaned_name = self._clean_drug_name(drug_ent["text"])
            if not self._is_valid_drug(cleaned_name):
                continue
            
            # Context filter: reject if drug name contains or is adjacent to non-medical words
            # Skip context filter for rule-based high-confidence matches (score >= 0.9)
            if drug_ent.get("score", 1.0) < 0.9:
                cleaned_name_lower = cleaned_name.lower()
                non_medical_keywords = ["brush", "paint", "tooth", "dental", "electric", "gum", "wash", "gate", "col", "oral", "colgate"]
                # Check if drug name itself contains non-medical keywords
                if any(keyword in cleaned_name_lower for keyword in non_medical_keywords):
                    continue
                # Check immediate context (10 chars before/after) - but be more permissive
                # Only filter if the drug is clearly part of a non-medical phrase
                context_before = text[max(0, drug_ent["start"]-10):drug_ent["start"]].lower()
                context_after = text[drug_ent["end"]:min(len(text), drug_ent["end"]+10)].lower()
                # Only reject if context strongly suggests non-medical (e.g., "tooth brush")
                if any(keyword in context_before and keyword in context_after for keyword in non_medical_keywords):
                    continue

            # Define boundary: segment extends from this drug until the next drug
            next_drug_idx = drug_indices[idx + 1] if idx + 1 < len(drug_indices) else len(entities)
            segment = entities[drug_i:next_drug_idx]

            med = {
                "drug": cleaned_name,     # frontend key is 'drug', not 'name'
                "name": cleaned_name,     # keep 'name' as alias for compatibility
                "strength": None,
                "dosage": None,
                "frequency": None,
                "duration": None,
                "route": None,
                "form": None
            }

            for e in segment:
                lbl = e["label"]
                etxt = e["text"].strip()
                # Use precise rules to parse sub-components from the entity text
                parsed = self._parse_medication_line(etxt)
                
                # STRENGTH: "500 mg", "250mg" etc
                if lbl == "STRENGTH" and not med["strength"]:
                    med["strength"] = etxt
                elif not med["strength"] and parsed["strength"]:
                    med["strength"] = parsed["strength"]

                # DOSAGE: "1 tablet", "2 caps"
                if lbl == "DOSAGE" and not med["dosage"]:
                    med["dosage"] = etxt
                
                if not med["frequency"]:
                    med["frequency"] = parsed["frequency"] or (etxt if lbl == "FREQUENCY" else None)
                if not med["duration"]:
                    med["duration"] = parsed["duration"] or (etxt if lbl == "DURATION" else None)
                if not med["route"]:
                    med["route"] = parsed["route"] or (etxt if lbl == "ROUTE" else None)
                if not med["form"] and lbl == "FORM":
                    med["form"] = etxt

            # STRICT VALIDATION: Must have at least a frequency OR a duration OR a dosage to be a valid prescription line
            # RELAXED: For very fragmented OCR, accept drug name alone if it's long enough (>= 6 chars)
            if med["frequency"] or med["duration"] or med["dosage"] or len(cleaned_name) >= 6:
                meds.append(med)

        return meds

        # Separate combined expressions (Strength and Frequency)
        text = re.sub(r'(\d+\s*mg)\s*(\d-\d-\d)', r'\1 |FREQ| \2', text)
        
        # Normalize dosage patterns (e.g., 1 - 0 - 1 -> 1-0-1)
        text = re.sub(r'(\d)\s*-\s*(\d)\s*-\s*(\d)', r'\1-\2-\3', text)
        
        # Fix duration shorthand (e.g., x 5days -> 5 days)
        text = re.sub(r'x\s*(\d+)\s*(days?|weeks?|months?)', r'\1 \2', text)
        
        # Remove junk characters (keep only alphanumeric, hyphens, and spaces)
        text = re.sub(r'[^a-zA-Z0-9\-\s]', ' ', text)
        
        # Remove tiny junk tokens (1-2 chars) that are not numbers
        text = re.sub(r'\b[a-zA-Z]{1,2}\b', ' ', text)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _clean_ocr_text(self, text: str) -> str:
        """
        Normalize raw OCR text before NER processing.
        Fixes merged tokens, units, and dosage patterns.
        """
        # Fix merged tokens (CamelCase to Space)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Remove garbage artifacts (like ##)
        text = re.sub(r'##+', '', text)
        
        # Fix units (e.g., 500mg -> 500 mg)
        text = re.sub(r'(\d+)\s*(mg|ml|g|mcg|tabs|caps)', r'\1 \2', text)
        
        # Separate combined expressions (Strength and Frequency)
        text = re.sub(r'(\d+\s*mg)\s*(\d-\d-\d)', r'\1 |FREQ| \2', text)
        
        # Normalize dosage patterns (e.g., 1 - 0 - 1 -> 1-0-1)
        text = re.sub(r'(\d)\s*-\s*(\d)\s*-\s*(\d)', r'\1-\2-\3', text)
        
        # Fix duration shorthand (e.g., x 5days -> 5 days)
        text = re.sub(r'x\s*(\d+)\s*(days?|weeks?|months?)', r'\1 \2', text)
        
        # Remove junk characters (keep only alphanumeric, hyphens, and spaces)
        text = re.sub(r'[^a-zA-Z0-9\-\s]', ' ', text)
        
        # Remove tiny junk tokens (1-2 chars) that are not numbers
        text = re.sub(r'\b[a-zA-Z]{1,2}\b', ' ', text)
        
        return text

    def extract_entities(self, text: str) -> list:
        # Layer 0: Preprocess (learned corrections + shorthand expansion)
        text = self._preprocess(text)
        
        results = []
        
        # Layer 1: Med7 (Medication primary)
        doc_med7 = self.med7(text)
        results.extend([self._format_ent(ent) for ent in doc_med7.ents])
        
        # Layer 2: SciSpacy BioNLP (Anatomy and Chemicals)
        doc_sci = self.sci(text)
        for ent in doc_sci.ents:
            if ent.label_ in ["ANATOMY", "CHEMICAL", "ORGANISM"]:
                results.append(self._format_ent(ent))
        
        # Layer 2b: BC5CDR (Diseases + Chemicals)
        if self.sci_bc5cdr:
            doc_bc5cdr = self.sci_bc5cdr(text)
            for ent in doc_bc5cdr.ents:
                results.append(self._format_ent(ent))
        
        # Layer 3: Rule-based medical patterns
        results.extend(self._extract_medical_patterns(text))
        
        # Layer 4: Transformer NER (Secondary)
        if self.transformer_ner:
            try:
                trans_ents = self.transformer_ner(text)
                for ent in trans_ents:
                    label = self.ENTITY_MAP.get(ent['entity_group'], ent['entity_group'])
                    results.append({
                        "text": ent['word'],
                        "label": label,
                        "start": ent['start'],
                        "end": ent['end'],
                        "score": float(ent['score']),
                        "color": self.color_map.get(label, "#f0f0f0")
                    })
            except Exception as e:
                logger.warning(f"Transformer extraction failed: {e}")
        
        # Layer 5a: Apply learned entities from DB
        for term, lbl in self.learned_entities.items():
            for m in re.finditer(re.escape(term), text, re.IGNORECASE):
                results.append({
                    "text": text[m.start():m.end()],
                    "label": lbl,
                    "start": m.start(),
                    "end": m.end(),
                    "score": 1.0,
                    "color": self.color_map.get(lbl, "#f0f0f0")
                })
        
        # Layer 5b: Merge adjacent DRUG fragments
        results = self._merge_adjacent_fragments(results, text)
        
        # Layer 5c: Clean and Correct
        results = self._clean_entities(results)
        
        # Layer 5d: Split mixed entities
        results = self._split_mixed_entities(results)
        
        # Layer 5e: Filter OCR Noise
        results = self._filter_noise(results)
        
        # Layer 5f: Normalize entities
        results = self._normalize_entities(results)
        
        # Layer 5g: Deduplicate
        results = self._deduplicate_entities(results)
        
        return sorted(results, key=lambda x: x['start'])

    def _format_ent(self, ent):
        return {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "color": self.color_map.get(ent.label_, "#f0f0f0")
        }

    def _deduplicate_entities(self, entities: list) -> list:
        """
        Clean deduplication: remove exact duplicates and overlapping entities.
        Keeps longer entities when they overlap.
        Also removes duplicate drug names at different positions.
        """
        if not entities:
            return []
        
        # Sort by start position, then by length (longer first)
        sorted_ents = sorted(entities, key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        result = []
        seen_text_labels = {}  # Track (text.lower(), label) -> start position
        
        for ent in sorted_ents:
            # Check if this entity overlaps with any already kept entity
            overlaps = False
            for kept in result:
                # Check for overlap
                if not (ent['end'] <= kept['start'] or ent['start'] >= kept['end']):
                    # Overlapping - skip this one (kept is longer due to sort order)
                    overlaps = True
                    break
            
            if overlaps:
                continue
            
            # Check for duplicate text+label combinations (same drug at different positions)
            ent_key = (ent['text'].lower().strip(), ent['label'])
            if ent_key in seen_text_labels:
                # Skip duplicate - keep the first occurrence
                continue
            
            seen_text_labels[ent_key] = ent['start']
            result.append(ent)

        return result
    
    def _generate_standardized_text(self, medications: list) -> str:
        """
        Generate a clinically professional multiline prescription summary.
        """
        if not medications:
            return ""
        
        lines = []
        for med in medications:
            # Support both 'drug' (new) and 'name' (legacy) keys
            drug_name = med.get('drug') or med.get('name', 'Unknown Medication')
            line = drug_name
            
            # Append strength
            if med.get('strength'):
                line += f" {med['strength']}"
            
            # Append dosage
            if med.get('dosage'):
                line += f" {med['dosage']}"

            # Append form if available
            if med.get('form'):
                line += f" ({med['form']})"
                
            # Append frequency
            if med.get('frequency'):
                line += f" - {med['frequency']}"
            
            # Append duration with 'for'
            if med.get('duration'):
                line += f" for {med['duration']}"
                
            # Append route with 'via'
            if med.get('route'):
                line += f" via {med['route']}"
                
            lines.append(line)
        
        return "\n".join(lines)
    
    def extract_clinical_summary(self, text: str):
        # 0. Advanced OCR cleanup with corrections dictionary (CRITICAL)
        text = self._advanced_ocr_cleanup(text)
        # 0b. Reconstruct broken words
        text = self._fix_broken_words(text)
        
        entities = self.extract_entities(text)
        
        # Rule-based patterns are already added inside extract_entities(text) 
        # in the current version, but we'll ensure they are integrated for the summary grouping.
        medications = self._group_medications(entities, text)
        standardized_text = self._generate_standardized_text(medications)
        
        return {
            "entities": entities,
            "medications": medications,
            "medication_count": len(medications),
            "standardized_text": standardized_text
        }
    
    def resolve_medical_terms(self, text: str, entities: list = None) -> list:
        """
        Resolve medical terms using fuzzy matching and normalization.
        Returns list of resolved terms with confidence scores.
        """
        return []  # Placeholder - fuzzy matching disabled

medtex_engine = MedTexEngine()