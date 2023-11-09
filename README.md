# Artist-Explorer
Simple program to explore similar artists based on Spotify 'related' data.

To use the app you need access to Spotify API.

Guide how to get Spotify API access:

1. Make an account on https://developer.spotify.com/
2. Go to 'Dashboard' tab, under your profile name 
3. Create app. Put 'http://localhost:8080/callback' in Redirect URI* and check Web API checkbox.
4. From Dashboard tab go to the created app and go to the settings.
5. Copy Client ID and Client secret from the website to the config.py file
6. Run main.py

*If you want different localhost port remember to change it in config.py file too.