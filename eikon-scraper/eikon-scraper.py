import eikon as ek
import os

EIKON_API_KEY = os.getenv('EIKON_API_KEY')

print(EIKON_API_KEY)
ek.set_app_key(EIKON_API_KEY)
headlines = ek.get_news_headlines('EU AND POL', 1)
story_id = headlines.iat[0,2]
ek.get_news_story(story_id)
