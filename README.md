# Language Dectection test
Test of the use of unigrams and bigrams relative frequency for detecting language.
The algorithm implemented performs a comparison between the relative frequency in the trained data and the one in the given text, using a Pearson's correlation coefficient.

### Usage
##### Training
```bash
python langdetector.py train <Ngram-size> <training-data-dir>
```
Will create the weights/frequency files needed for detection.

##### Testing detection
```bash
python langdetector.py detect <Ngram-size> <test-file> <solutions-file>
```
Will perform the test and output the accuracy obtained as well as the lines where the algorithm got it wrong.

```bash
python langdetector.py langdetect <test-file> <solutions-file>
```

Will perform the test, but this time using the `langdetect` python module for comparision.