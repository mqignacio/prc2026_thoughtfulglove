# Performance Review Commission (PRC) Data Challenge 2026

- https://ansperformance.eu/study/data-challenge/dc2026/
- https://ansperformance.eu/study/data-challenge/dc2026/rationale.html
- https://ansperformance.eu/study/data-challenge/dc2026/data.html
- https://prc-data-challenge-2026.netlify.app/ranking.html

## OpenSky Network API Documentation

- https://openskynetwork.github.io/opensky-api/index.html
- https://openskynetwork.github.io/opensky-api/rest.html
- https://openskynetwork.github.io/opensky-api/trino.html
- https://openskynetwork.github.io/opensky-api/python.html

## Data access

- Use your OpenSky account credentials to access competition datasets and upload submissions to your team's bucket here -Important: In order to login you must click Other Authentication Methods and select Login with SSO. You will then be redirected to our Keycloak IAM where you can log in with your OpenSky Credentials.

## Submissions

- Submissions must be named `thoughtful-glove_vN.parquet` (where N is the version number of the submission). For example thoughtful-glove_v1.parquet If there are no problems with your submission you should see a result file appear in the bucket soon after.

NOTES:
- Team already created, named `thoughtful-glove`.
- Submission bucket: `prc-2026-thoughtful-glove`
- OpenSky account (`mqignacio`) already created:
    - Client ID: `mqignacio-api-client`
    - API Access: Standard (4,000 credits per day. This is your daily allowance, not the number of credits remaining. Live usage is returned in the `X-Rate-Limit-Remaining` response header.)
    - Credentials: `resources/credentials.json`