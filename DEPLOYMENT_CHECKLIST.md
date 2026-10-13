# Equinox release checklist

No commit, push, or deployment has been performed.

## Required before release

- Rotate the Reddit API secret previously embedded in source; treat it as exposed.
- Configure `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `EQUINOX_SECRET_KEY`, `EQUINOX_DEMO_PASSWORD`, and `EQUINOX_ENV=production` in the server environment, not Git.
- The current shared-password login is demo authentication, not production-grade identity management. Do not use it for real patient accounts or sensitive clinical data.
- Verify the actual EC2 service manager, project path, Python version, environment file location, reverse proxy, and restart command before modifying the host. No deployment automation or service definition is present in this repository.
- Back up the currently deployed revision and runtime configuration, keeping secrets out of Git.
- Stage only approved application files. Exclude local rollback backups, uploads/data, virtual environments, logs, and secret files.
- Test on a staging copy using the server's Linux/Python environment.

## Local checks performed

- Python syntax compilation for `app.py` and `aura.py`.
- `git diff --check`.
- Flask test-client checks: homepage/logo response, unauthenticated API denial, and cross-profile API denial.

These are not a substitute for browser testing, full regression tests, or staging deployment.

## Release procedure after approval

1. Stage only approved files and inspect the staged diff.
2. Run tests and review the staged diff again.
3. Commit and push the approved release.
4. Back up the live revision on EC2.
5. Deploy using the verified service procedure and configure environment variables outside Git.
6. Restart the verified application service.
7. Check homepage, login, user dashboard APIs, therapist search, patient detail, and logo in a real browser; inspect logs.
8. If checks fail, restore the backup and restart the service.

## Security follow-up

Before handling real users or sensitive data, replace shared-password login with individual accounts and password hashing; add CSRF protection, secure session-cookie settings, rate limiting, and a privacy/authorization review.
