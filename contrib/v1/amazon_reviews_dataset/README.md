# Find label errors in text data (example Amazon Reviews)

We released pre-prepared version of the Amazon5core reviews dataset.
Download it here: https://github.com/cgnorthcutt/label-errors/releases/tag/amazon-reviews-dataset

From the Amazon 5core dataset (40+ million examples), we select only the data that adheres to:
1. non-empty reviews.
2. label must be 1 star, 3 stars, or 5 stars. (2 and 4 star reviews are removed)
3. Only consider reviews with more than upvotes than downvotes (and at least one upvote).

You should have about 10 million examples left-over. These are higher quality, which will allow us to have more control over noise in the labels (instead of just general noise in the text itself).

The dataset has been formatted in [fastext format](https://fasttext.cc/docs/en/supervised-tutorial.html#getting-and-preparing-the-data) for you. Here are the first two lines of my formatted training data file:

```
__label__5 I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!
__label__4 This work bears deep connections to themes first explored in Tad Williams' original breakout novel, about a brave young cat who travels to an underground netherworld to face an ancient evil.  As the owner of two cats myself, after the second read-through, I realized that this novel has much to teach about the critical importance of dealing with fur and dust.I could only give four stars, though, because the cats do not agree, and indeed wish I had not made this purchase.
```

When training, we pre-process the training data as follows:

```bash
cat amazon5core.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > amazon5core.preprocessed.txt
```

## Now you're ready to use cleanlab / confident learning.

Use the scripts here, in this order:

  1. [amazon_pyx.ipynb](/examples/amazon_reviews_dataset/amazon_pyx.ipynb)
      * See [amazon_pyx_tfidf.ipynb](/examples/amazon_reviews_dataset/amazon_pyx_tfidf.ipynb) for an example with no fasttext dependency.
  2. [amazon_label_errors.ipynb](/examples/amazon_reviews_dataset/amazon_label_errors.ipynb)
  3. [compare_cl_vs_vanilla.ipynb](/examples/amazon_reviews_dataset/compare_cl_vs_vanilla.ipynb)
  4. [cl_vs_vanilla_analysis.ipynb](/examples/amazon_reviews_dataset/cl_vs_vanilla_analysis.ipynb)
