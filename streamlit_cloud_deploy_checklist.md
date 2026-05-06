# GigShield Streamlit Cloud Deploy Checklist

This checklist is for deploying the GigShield Risk Simulator to Streamlit Community Cloud and embedding it into WordPress with an iframe.

## 1. Files to Upload to GitHub

Make sure these files are included in your GitHub repository:

- `streamlit_app.py`
- `Main_program.py`
- `weather_service.py`
- `geocode_service.py`
- `osm_service.py`
- `district_lookup.py`
- `requirements.txt`
- `xgb_risk_score.json`
- `xgb_est_time.json`
- `xgb_feature_meta.pkl`
- `google_geocode_cache.json`
- `osm_signal_cache.json`

If your district lookup depends on a local boundary file, also include:

- `hk_district_boundaries.geojson`

If your current code uses another local boundary file name, upload that exact file name.

## 2. Important Security Note

Do not hard-code your real Google Maps API key in a public GitHub repository.

Recommended approach:

1. Remove the real key from code.
2. Use Streamlit secrets instead.

Example Streamlit secret:

```toml
GOOGLE_MAPS_API_KEY = "your_actual_google_maps_api_key"
```

## 3. Recommended Code Adjustment

Your current main program mainly reads the Google API key from:

- local constant
- `os.getenv("GOOGLE_MAPS_API_KEY")`

For Streamlit Cloud, it is safer to support both:

- `st.secrets["GOOGLE_MAPS_API_KEY"]`
- `os.getenv("GOOGLE_MAPS_API_KEY")`

If needed, update the code so Streamlit Cloud can read the secret correctly.

## 4. Create the GitHub Repository

1. Create a new GitHub repository.
2. Upload all required files.
3. Confirm that `streamlit_app.py` is in the repository root, or note its exact path.

## 5. Deploy to Streamlit Community Cloud

Go to:

- [https://share.streamlit.io/](https://share.streamlit.io/)

Then:

1. Sign in with GitHub.
2. Click `New app`.
3. Select your repository.
4. Set the main file path to:

```text
streamlit_app.py
```

5. Click `Deploy`.

After deployment, Streamlit will generate a public URL, for example:

```text
https://gigshield-risk-simulator.streamlit.app
```

## 6. Add Streamlit Secrets

In Streamlit Cloud:

1. Open your deployed app settings.
2. Find the `Secrets` section.
3. Add:

```toml
GOOGLE_MAPS_API_KEY = "your_actual_google_maps_api_key"
```

4. Save and reboot the app if needed.

## 7. Test the Streamlit App First

Before embedding into WordPress, confirm the Streamlit app works directly:

- page loads successfully
- route calculation works
- Google Directions works
- weather output works
- XGBoost model loads correctly

## 8. Embed into WordPress

Use the iframe code in the separate HTML file:

- `wordpress_iframe_embed_code.html`

In WordPress:

1. Edit the target page.
2. Add a `Custom HTML` block.
3. Paste the iframe code.
4. Replace the placeholder Streamlit URL with your actual deployed URL.

## 9. If iframe Does Not Display

Possible reasons:

- Streamlit site blocks iframe embedding
- WordPress security plugin blocks iframe
- wrong app URL

Fallback option:

Use a button or link to open the Streamlit app in a new tab instead of embedding.

## 10. Final Demo Checklist

Before presentation, confirm:

- GitHub repo is complete
- Streamlit app is deployed
- Google API key works in Streamlit Cloud
- iframe page loads in WordPress
- at least 2 to 3 test routes run successfully
- backup plan is ready:
  - direct Streamlit URL
  - WordPress button link

## 11. Recommended Architecture Sentence for Report or Viva

You may describe the deployment approach like this:

```text
WordPress is used as the front-end presentation layer, while the GigShield Python risk engine is deployed separately as a Streamlit-based application. The WordPress page embeds the simulator through an iframe, allowing users to interact with the AI-driven route risk assessment interface within a website environment.
```
