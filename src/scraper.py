import requests
from bs4 import BeautifulSoup
import urllib.request
import urllib.error
from pprint import pprint
from html_table_parser.parser import HTMLTableParser
import pandas as pd


# Opens a website and read its
# binary contents (HTTP Response Body)
def url_get_contents(url):

    # Opens a website and read its
    # binary contents (HTTP Response Body)

    try:
        # Send an HTTP GET request to the URL
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)

        # Check the status code of the response
        if response.status == 200:
            # The request was successful
            html = response.read()
            # Do something with the HTML content
        elif response.status == 301:
            # The URL has permanently moved
            new_url = response.getheader('Location')
            print('The URL has moved to:', new_url)
            # Send a new request to the new URL
            req = urllib.request.Request(new_url)
            response = urllib.request.urlopen(req)
            html = response.read()
            # Do something with the HTML content
        else:
            # The request failed for some other reason
            raise RuntimeError('Error code: ', response.status)

    except urllib.error.HTTPError as e:
        raise RuntimeError('HTTP error: ', e.code)
    except urllib.error.URLError as e:
        raise RuntimeError('URL error: ', e.reason)
    except Exception as e:
        raise RuntimeError('Error occurred: ', e)

    #reading contents of the website
    return html

def scrape(url):
    # defining the html contents of a URL.
    xhtml = url_get_contents(url).decode('utf-8')

    # Defining the HTMLTableParser object
    p = HTMLTableParser()

    # feeding the html contents in the
    # HTMLTableParser object
    p.feed(xhtml)

    # Now finally obtaining the data of
    # the table required
    pprint(p.tables[1])

    # converting the parsed data to
    # dataframe
    print("\n\nPANDAS DATAFRAME\n")
    print(pd.DataFrame(p.tables[1]))


def soup_scrape(url):
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
