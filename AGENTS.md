# PROJECT GIT AND README POLICY

Before performing work that changes this repository, read this file and the existing root `README.md`.

1. The root README.md is the single authoritative GitHub project homepage and development changelog.
2. Every meaningful Git commit MUST update the existing root README.md in the SAME commit.
3. Record the actual update time in UTC+08:00, the operation performed, affected files, and the verification status.
4. Keep the existing changelog history. Put the newest entry first.
5. Update Mermaid workflow diagrams whenever actual program logic or data flow changes.
6. Update algorithm parameters and descriptions whenever the implementation changes.
7. Never claim that an experiment was executed or verified without supporting evidence.
8. Do not create duplicate README versions.
9. No meaningless or empty commits.
10. Audit, commit, push, and verify the remote SHA.
11. Follow the repository's existing Git safety policy. Never force-push, rewrite history, discard existing changes, or disable TLS verification without explicit authorization.

Do not install hooks that modify files or intercept Git operations unless the user explicitly authorizes them.
