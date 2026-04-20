import spacy

class MedTexEngine:
    def __init__(self):
        # Model 1: Med7 for Medication details
        self.med7 = spacy.load("en_core_med7_lg")
        # Model 2: SciSpacy for Anatomy and Chemicals
        self.sci = spacy.load("en_ner_bionlp13cg_md")
        
        self.color_map = {
            # Med7 Labels
            "DRUG": "#ffcfcc", "DOSAGE": "#d3f9d8", "DURATION": "#fff3bf",
            "FORM": "#eebefa", "FREQUENCY": "#d0ebff", "ROUTE": "#fff4e6",
            # SciSpacy Labels
            "ANATOMY": "#c5f6fa", "CHEMICAL": "#ffec99", "CANCER": "#ffc9c9"
        }

    def extract_entities(self, text: str):
        # Process with both models
        doc_med7 = self.med7(text)
        doc_sci = self.sci(text)
        
        results = []
        
        # Extract from Med7
        for ent in doc_med7.ents:
            results.append(self._format_ent(ent))
            
        # Extract from SciSpacy (Filtering for Anatomy and Chemicals)
        for ent in doc_sci.ents:
            if ent.label_ in ["ANATOMY", "CHEMICAL", "ORGANISM"]:
                # Avoid duplicates if Med7 already caught it
                if not any(r['start'] == ent.start_char for r in results):
                    results.append(self._format_ent(ent))
        
        # Sort by start position for the frontend
        return sorted(results, key=lambda x: x['start'])

    def _format_ent(self, ent):
        return {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "color": self.color_map.get(ent.label_, "#f0f0f0")
        }

medtex_engine = MedTexEngine()