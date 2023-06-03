import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import datetime
import os
import argparse
from sentence_transformers import SentenceTransformer, util

pd.set_option('display.max_rows', 1200)

parser = argparse.ArgumentParser(description='Semantle!')

parser.add_argument('--random', metavar='r', type=bool, default=False ,
                    help='Generate a random word instead of the daily!', action=argparse.BooleanOptionalAction
)
args = parser.parse_args()#sys.argv[2:])

model = SentenceTransformer('all-MiniLM-L6-v2')

# TODO a much longer word list
# TODO a hint!!!! when the word is IMPOSSIBLE
# TODO generate given a word!
fi = "google-10000-english/google-10000-english.txt"
with open(fi) as file:
  englist = file.read().splitlines()

englist = np.array(englist)
fi = "transformer.tch"
if os.path.isfile(fi):
  print("Loading Transformer Model This may take a second...")
  engbeddings = torch.load(fi)
else:
  print("Generating Transformer Model for the first time, this may take a few minutes...")
  engbeddings = model.encode(englist, convert_to_tensor=True)
  torch.save(engbeddings, fi)

# use today's date as a random key to pick a word -- this won't vary across computer calendars
if (args.random == False):
  seed = int(datetime.datetime.today().strftime("%Y%m%d"))
  random.seed(seed)
  np.random.seed(seed)
# use this to make a random ordering
# compute days since time 0
# pick that element in the order
w1 = random.choice(englist)

e1 = model.encode(w1, convert_to_tensor=True)
w1dist = util.cos_sim(engbeddings, e1)
simlist = pd.DataFrame({"word": englist, "score":w1dist[:,0].numpy()})
simlist = simlist.sort_values("score", ascending=False)
simlist["score2"] = np.power(simlist["score"], 2)
simlist["eff_score"] = np.power(simlist["score"], 2)*100
simlist["rank"] = np.zeros(simlist.shape[0]).astype(int)
simlist["full_rank"] = np.arange(simlist.shape[0])[::-1]
simlist.loc[(simlist["full_rank"] < 9999) & (simlist["full_rank"] > 8998), "rank"] = np.arange(1000)[::-1]
score = np.array(simlist["score"])
guesses = pd.DataFrame(columns = ['guess', 'word', 'score', 'rank'])

# get top 1, 10, 1000 guess numbers
print("----------------------------------------------------------------------------------------")
print("You're playing Brian's new TRANSFORMED semantle! \nHere are some scores to start you off")
if (args.random==False):
  print("This is the DAILY semantle!")
else:
  print("This is a RANDOM semantle!")
print("----------------------------------------------------------------------------------------")
print("Nearest: {:.2f} \n10th:    {:.2f} \n1000th:  {:.2f}".format(simlist['eff_score'].iloc[1], simlist['eff_score'].iloc[11],simlist['eff_score'].iloc[1000]))

while True:
  # get user input word
  w2 = input("Guess:")
  e2 = model.encode(w2, convert_to_tensor=True)
  # report score
  dist = util.cos_sim(e1, e2)
  fdist = torch.pow(dist, 2)
  sdist = "{:.2f}".format(fdist.numpy()[0][0]*100)
  # if it's top 1000, print (approx)ranking
  rank = (score > dist.numpy()).argmin()
  if rank < 1000:
    srank = F"{str(1000-rank)}/1000"
  else:
    srank = ""
  # if it's not in our 10k words, add ???
  if (simlist["word"] == w2).sum() < 1:
    known = "???"
  else:
    known = ""
  # if it's correct, congratulations, print everything
  if (w2 not in np.array(guesses["word"])):
    guesses = guesses._append({'guess' : guesses.shape[0], 'word' : w2, 'score' : float(sdist), 'rank' : srank},
          ignore_index = True)
    guesses = guesses.sort_values("score", ascending=False)

  # add this guess to a table and keep it sorted
  if (dist > 0.9999999):
    print(F"CONGRATULATIONS! You got that in             {guesses.shape[0]}     guesses!")
    print("Today's totally not madeup world average was {:.2f} guesses".format(np.random.geometric(p=(1.1/35.0),size=9).mean()))
    print(guesses.to_string(index=False))
    print("--------------------------------------------------------------------------------")
    nearest = input("Would you like to see the nearest word list?")
    
    if True:#"y" in nearest.lower():
      print(simlist.iloc[1:1000][["word", "eff_score", "rank"]])
      print("--------------------------------------------------------------------------------")
      print(F"CONGRATULATIONS! You got that in             {guesses.shape[0]}     guesses!")
      print("Today's totally not madeup world average was {:.2f} guesses".format(np.random.geometric(p=(1.1/35.0),size=9).mean()))


  

      break
    else:
      break
  else:
    # TODO : flip and print all guesses so nearest are at the bottom but you can scroll up
    print(guesses.head(20).to_string(index=False))
    print(F"\t\t {w2} : \t{sdist} \t {srank} {known}")


