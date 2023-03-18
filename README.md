# Artist-Explorer
Small app based on Spotify API made to explore artists

# How to run it*:
To run this app you need Spotify dev account, so:
1. Go to https://developer.spotify.com/dashboard/login
2. Log in with your Spotify account
3. Create new app by clicking 'Create an app'
4. Fill name and description with anything you like
5. In app overview click edit settings and to 'Redirect URIs' add "http://localhost:8080/callback" **
6. Save changes and go to config.py file
7. From app overview copy Client ID and paste it as string as SPOTIPY_CLIENT_ID value in config.py file
	it should look like this: SPOTIPY_CLIENT_ID = 'aa11aa'
8. From app overview click 'Show client secret' under your ID and paste it as string as SPOTIPY_CLIENT_SECRET value in config.py file
9. Done. You're good to run main.py

* I assume you dont have an spotify dev account and app
** In case you have localport 8080 occupied select different port and change SPOTIPY_CLIENT_URL value in config.py file
