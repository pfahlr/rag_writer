
import crossref_commons.retrieval
import json 
from crossref_commons.config import API_URL 
import urllib
import argparse 
import re
import sys
import requests

def query_generic(searchtext):
  print(searchtext)
  query = urllib.parse.urlencode([('query', searchtext)])
  print(query)
  # search for an article using a totally unsophisticated plaintext search
  #query = searchtext
  response = requests.get(f"{API_URL}/v1/works?{query}")
  if response.ok:
    results = json.loads(response.content)
    return results
  else: 
    return []

def get_json_by_doi(doi):
  # get a whole bunch of metadata about a published work using the DOI
  return crossref_commons.retrieval.get_publication_as_json(doi)


def get_doi_or_nothing(start_year, end_year, title, author, publication):
  results = [{'doi':'10.123456/abcdef.thisisntimplementedyet.zyxwvut/987654321'}]
  # more than 1 indicates not specific enough to match accurately, less than 1 indicates nothing found - possibly too specific query.
  # the idea behind this function is it should not require editorial review, if you're not getting a result here, try searching using 
  # using the more broad query, but you'll likely need to review manually to find the 1<->1 match. 
  if len(results) == 1: 
    return results[0]['doi']
  else:
    return None
  
def main():
  parser = argparse.ArgumentParser(description="A script to query the crossref.org database: the organization responsible for the assignment of DOI numbers to journal articles.")
  parser.add_argument('--doi', '-d', type=str, help="the DOI number formatted as 10.1234/xyyzz/1233444, calling the script with this arguement will query by DOI for all fields")
  parser.add_argument('--searchtext', '-s', type=str, help="a plaintext search string, calling the script with this argument will query the database for records matching the full text string and return all fields presuming the caller will need to look through the results to find the result they're looking for")
  parser.add_argument('--find-doi', '-fd', action='store_true', help="call the script with --author, --title, --year-range and --publication fields. If the the query returns a result set of size 1, the DOI will be returned directly")
  
  for arg in sys.argv:
    if arg == '--find-doi' or arg == '-fd':
      parser.add_argument('--title', '-t', type=str, required=True, help="a string representing the title, this argument is used in conjunction with other arguments to create a targeted search that should ideally return one result")
      parser.add_argument('--author', '-a', type=str, default="", help="a string representing the a range of years formatted as 2012-2013 or 2019:2025, this argument is used in conj...")
      parser.add_argument('--year-range', '-yr', type=str, required=True, help="a string representing the author, this argument is used in conj...")
      parser.add_argument('--publication', '-p', type=str, default="", help="a string representing the author, this argument is used in conj...")
      break
  
  args = parser.parse_args()

  if args.find_doi: 
    splitregex = r"[-:|,\.]"
    items = re.split(splitregex, str(args.year_range))
    start_year = items[0]
    end_year = items[1]
    publication = str(args.publication)
    author = str(args.author)
    title = str(args.title)
    result = get_doi_or_nothing(start_year, end_year, title, author, publication)
    print(result)
    exit(1)

  else: 
    if args.doi: 
        result = get_json_by_doi(str(args.doi))
        print(result)
        exit(1)
    elif args.searchtext: 
        result = query_generic(str(args.searchtext))
        print(result)
        exit(1)
  
if __name__ == "__main__":
    main()