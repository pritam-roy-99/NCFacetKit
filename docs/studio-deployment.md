# Research Studio deployment

## Public GitHub Pages frontend

The frontend is published from the user's GitHub Pages repository:

- Site: https://pritam-roy-99.github.io/NCFacetKit/
- Source: https://github.com/pritam-roy-99/pritam-roy-99.github.io/tree/main/NCFacetKit

The frontend is a **static API client**. It is not a Python execution host. It does not calculate facets, see-saw or moment-SDP results by itself and does not show mock results.

## Python compute backend

The complete v0.5.0 FastAPI/quantum implementation was prepared in the Research Studio distribution archive. It must be added to this repository and deployed as a separate Python web service. The deployment archive includes `Dockerfile` and `render.yaml`.

After deploying the Python backend to Render or a comparable host:

1. Verify `GET https://YOUR-SERVICE/api/health` reports `status: ok`.
2. Configure FastAPI CORS to accept the exact frontend origin `https://pritam-roy-99.github.io`. In FastAPI, use `CORSMiddleware` with that value in `allow_origins`, `allow_methods=['GET','POST']`, and `allow_headers=['Content-Type']`.
3. Open the GitHub Pages frontend, enter the backend HTTPS URL, and click **Connect server**. The frontend stores that URL locally in the browser.
4. Run a small test first: four preparations, two binary measurements, `(P0+P1)/2=(P2+P3)/2`, and the classical bound for `p(0|1,0)-p(0|2,0)-p(0|0,1)+p(0|2,1)` should equal 1.

For public access, configure authentication/rate limits and resource limits on the backend; expensive facet enumeration and semidefinite programs can consume substantial CPU/RAM.

Important: **GitHub Pages cannot execute CVXPY, Python, FastAPI, or an SDP solver.** Until the backend is online and connected, the deployed frontend will show a disconnected-server message.
