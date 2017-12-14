# homonyms
Detecting homonyms in text and finding their best translations.

Uses fasttext vectors, gensim, sklearn.

Assumes you have en and ru fasttext vectors downloaded into a `vec` folder in the root directory.

Data breakdown by number of samples, across 4 orders of magnitude

```
  448308 total
  169117 bank.txt
  108071 club.txt
   96666 match.txt
   22020 bear.txt
   20683 volume.txt
   14100 mess.txt (non-homonym)
    9524 bat.txt
    5988 organ.txt
    1523 mint.txt
     616 stalk.txt
```