const API_URL = '';  // same origin as the page

// scope lucide to the toggle button only
const eyeToggle = document.getElementById('eyeToggle');
eyeToggle.addEventListener('click', function () {
    const input = document.getElementById('passwordInput');
    const isPassword = input.type === 'password';
    input.type = isPassword ? 'text' : 'password';
    // lucide replaces <i> with <svg>, so swap the whole icon node
    this.innerHTML = `<i data-lucide="${isPassword ? 'eye-off' : 'eye'}" aria-hidden="true"></i>`;
    this.setAttribute('aria-label', isPassword ? 'Hide password' : 'Show password');
    lucide.createIcons({ nodes: [this] });
});

// clear error on input
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
        errEl.textContent = 'Enter your email and password.';
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Signing in…';

    try {
        // do not send the role; the server determines role from credentials
        const res = await fetch(`${API_URL}/api/auth/signin`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        const data = await res.json();

        if (!res.ok) {
            errEl.textContent = res.status === 401
                ? 'That email and password do not match an account.'
                : (data.detail || 'Sign-in failed. Try again.');
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
        errEl.textContent = 'Could not reach the server. Check that it is running.';
        btn.disabled = false;
        btn.textContent = 'Sign in';
    }
});
