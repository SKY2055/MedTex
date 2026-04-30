import Dashboard from './components/Dashboard'
import Login from './components/Login'
import { AuthProvider, useAuth } from './context/AuthContext'

function AppContent() {
    const { user, loading } = useAuth();

    if (loading) {
        return (
            <div className="min-h-screen bg-slate-50 flex items-center justify-center">
                <div className="text-slate-500">Loading...</div>
            </div>
        );
    }

    return user ? <Dashboard /> : <Login />;
}

function App() {
    return (
        <AuthProvider>
            <AppContent />
        </AuthProvider>
    );
}

export default App
