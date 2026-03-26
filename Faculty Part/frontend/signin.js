// Password show/hide
document.getElementById('eyeToggle').addEventListener('click', function () {
    const input = document.getElementById('passwordInput');
    const isPassword = input.type === 'password';
    
    input.type = isPassword ? 'text' : 'password';
    
    const icon = this.querySelector('i');
    icon.setAttribute('data-lucide', isPassword ? 'eye-off' : 'eye');
    lucide.createIcons();
});

// Form submit
document.getElementById('signinForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const email    = document.getElementById('emailInput').value.trim();
    const password = document.getElementById('passwordInput').value;
    const role     = document.getElementById('roleSelect').value;
    const btn      = document.getElementById('signinBtn');
    const errEl    = document.getElementById('signinError');
    
    // Basic client-side validation
    errEl.textContent = '';
    
    if (!email || !password) {
        errEl.textContent = 'Please fill in all fields.';
        return;
    }
    
    btn.disabled = true;
    btn.textContent = 'Signing in…';
    
    try {
        const res = await fetch('/api/auth/signin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, role })
        });
        
        const data = await res.json();
        
        if (!res.ok) {
            errEl.textContent = data.error || 'Invalid credentials. Try again.';
            btn.disabled = false;
            btn.textContent = 'Sign in';
            return;
        }
        
        // Persist session
        sessionStorage.setItem('classai_token', data.token);
        sessionStorage.setItem('classai_role',  data.role);
        sessionStorage.setItem('classai_user',  JSON.stringify(data.user));
        
        // Redirect
        window.location.href = '/chat';
        
    } catch (err) {
        errEl.textContent = 'Network error. Check your connection.';
        btn.disabled = false;
        btn.textContent = 'Sign in';
    }
});

// Dev bypass
function bypassAuth() {
    sessionStorage.setItem('classai_token', 'dev-bypass-token');
    sessionStorage.setItem('classai_role', 'student');
    sessionStorage.setItem('classai_user', JSON.stringify({ name: 'Dev User', email: 'dev@nmims.edu' }));
    window.location.href = '/chat';
}
