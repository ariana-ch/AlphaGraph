Thank you; this is not quite what I have done so lets update. Here is a detailed description of what I ahve done so far.

My objective was to construct tradable portfolios that rebalance every 5 days and which contain approximately 40 stocks with the capping of the number of stocks not enforced but learned using an appropriate, differentiable loss functions. 
To do this I used price and volume data (structured data)
and news (unstructured data).
I first trained 3 models that used only price and volume data based on PatchTST, bidirectional LSTM and NBEATSx.
Thes models were trained using the last 30 days of data (prices, volumes and technical indicators constructed from the raw price and volume data, all with a daily frequency) and on 512 of the most liquid nasdaq stocks. 
The model were trained to take in a tensor of shape batch x 30 x 512 x d_features and return a tensor of shape 512 x 1 with the weights for each stock. These weights were used with the next 5 day returns (not log, raw) of each of the 512 stocks to compute the loss.
For the loss I have tried a few different options. Given that my objective was to have ~40 components in the portfolio I used a loss that penalised if the cardinality of the portfolio was higher or lower than 40. This function was designed to be differentiable. Initially I used the negative sharpe ratio with a target sharpe to optimise the sharpe but this was too unstable because each batch had a limited and relatively small number of holding periods. Instead I used a mean-variance utility with a risk aversion coefficient. I also have a loss term that controls volatility with a target volatility and a term that penalises for very small weights.

The training data was batch such that each batch had sequential prediction dates, i.e. the first entry in the batch was from t-30 to t-1, with holding period t to t+4, the next t-29 to t with holding period t+1 to t+5, to increase the number of training periods. For the testing and validation sets I used non-overlapping periods, i.e. 
t-30 to t-1 with holding from t to t+4 and then t-25 to t+4 with holding from t+5 to t+9 and so on.

Early stopping used the validation sharpe ratio.

Various metrics were used to assess the model including number of components, turnover, cumulative portfolio return, sharpe, max drawdown and calmar.

Phase two involved adding the news data. To do this I built a KG transformer.
The KGTransformer, inspired by EvoKG has a static embedding and a dynamic embedding with the dynamic embedding updated each day whereas the static is the same always but it is a parameter so it gets learned too. 
The architecture of this involves a structural and temporal component for each of the two embeddings. On each day I convolute using RGCN for the structural (combine the static and dynamic) and similarly for the temporal but the temporal also has an RNN (either RNN or GRU cell) to evolve day on day. I added a decay to the embeddings so that nodes that were seen in the past fade away in the dynamic embedding (they survive in the static and can be reintroduced through the RGCN if the those nodes are in the news again).

The output of the KGTransformer is an embedding of shape N_nodes x embedding_dim_for_graph. I then combined this with the price model output. 
Here the price model is one of the previous price models (but prior to being flattened to 512 x 1, I keep the hidden dim 512 x dim). To train the full model I iterate over news dates and do a forward pass, until I am aligned with the current prediction date. I then use cross attention between the Batch - size prediction dates and an aligned  batch-size x N_nodes x graph_embedding tensors with the price attending to the graph. The idea here is that the prices learn about the market from the news.

The resulting output is then passed through an MLP to get 512, 1 weights. Note that I always use entmax15 activation (I tried other activations too to enforce sparsity of weights). I wanted differentiability so I didn't want to have a hard cut.

I can't really use just the graph to construct a portfolio because I don't have enough companies in the news at any given time to get the necessary embeddings. This is a limitation of my news source but it would be an interesting avenue if I was able to do this.

I did also consider other architectures - I am not sure if this is relevant.

I can discuss more on the metrics and possible ablation studies that I can still perform - you can advice on this point.

Below I outline at high level some of what I have done, omitting a lot of the steps

1. Collected price data and volume data and news data from Jan 2020 to August 2025
2. Processed the articles and used Gemini to extract knowledge graph triplets using a well defined prompt and schema
3. From the price and volume data, I generated a set of 45 features, using technical indicators and different lags. The data is daily. 
4. Implemented and trained the price models and optimised
5. Implemented the KG transformer
6. Built the cross attention and trained the full model together

Studied the results looking at the test period and what stocks I am holding, how the portfolio evolves when does it do best etc

Based on this, can you please provide an updated structure?