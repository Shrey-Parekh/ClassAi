const API_URL = 'http://localhost:8001';

// A32: scope lucide to the toggle button only
const eyeToggle = document.getElementById('eyeToggle');
eyeToggle.addEventListener('click', function () {
    const input = document.getElementById('passwordInput');
    const isPassword = input.type === 'password';
    input.type = isPassword ? 'text' : 'password';
    const icon = this.querySelector('i');
    icon.setAttribute('data-lucide', isPassword ? 'eye-off' : 'eye');
    lucide.createIcons({ nodes: [this] });
});

// A30: clear error on input
['emailInput', 'passwordInput'].forEach(id => {
    document.getElementById(id).addEventListener('input', () => {
        document.getElementById('signinError').textContent = '';
    });
});

document.getElementById('signinForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const email    = document.getElementById('emailInput').value.trim();
    const password = document.getElementById('passwordInput').value;
    const btn      = document.getElementById('signinBtn');
    const errEl    = document.getElementById('signinError');

    errEl.textContent = '';

    if (!email || !password) {
        errEl.textContent = 'Please fill in all fields.';
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Signing in…';

    try {
        // A2: do NOT send role — server determines role from credentials
        const res = await fetch(`${API_URL}/api/auth/signin`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        const data = await res.json();

        if (!res.ok) {
            errEl.textContent = data.error || 'Invalid credentials. Try again.';
            btn.disabled = false;
            btn.textContent = 'Sign in';
            return;
        }

        // Role comes from server response (A2)
        sessionStorage.setItem('classai_token', data.token);
        sessionStorage.setItem('classai_role',  data.role);
        sessionStorage.setItem('classai_user',  JSON.stringify(data.user || {}));

        window.location.href = '/chat';

    } catch (err) {
        errEl.textContent = 'Network error. Check your connection.';
        btn.disabled = false;
        btn.textContent = 'Sign in';
    }
});
