#! /usr/bin/python3

# import bs4
# from bs4 import BeautifulSoup
import re
import xml.etree.ElementTree as ET
from lxml import etree
import lxml

files = ['club','bank','bat','bear','club','match','mess','mint','organ','stalk','volume']

for f in files:
    filepath ='./dat/en_ru/{}.xml'.format(f)
    g = open('./dat/en_ru/{}.preprocessed'.format(f),'w+')
    regex = re.compile(re.escape(f), re.IGNORECASE)

    tree = ET.parse(filepath)
    root = tree.getroot()
    table = root[1][0]
    try:
        for row in table.findall('{urn:schemas-microsoft-com:office:spreadsheet}Row')[1:]: # First row is a legend, can skip
            eng = row[-2][0].text
            rus = row[-1][0].text
            eng = regex.sub(f.lower(), eng)

            eng = eng.replace('\n',' ')
            rus = rus.replace('\n',' ')


            eng_text = eng.split('[')[0]
            rus_text = rus.split('[')[0]

            g.write('{}\n'.format(eng_text))
            # g.write('{}\n{}\n'.format(eng_text, rus_text))
    except Exception as e:
        print(e)
        print(eng)
        print(rus)