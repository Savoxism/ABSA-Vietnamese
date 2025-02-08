# ABSA-Vietnamese
 This is a complete reimplementation of the paper solving the problem of ABSA in Vietnamese language,



Why Use Both Binary Cross-Entropy and Softmax?
The key idea is that in Aspect Category Sentiment Analysis (ACSA), each Aspect#Category,Polarity is like an independent binary classification problem. However, rather than treating them as completely separate, we use Softmax because sentiments are not independent—they have an implicit relationship.

Mutual Exclusivity of Sentiments:
If an Aspect#Category is strongly positive, it is less likely to be negative or neutral.
If an Aspect#Category is very negative, it is unlikely to be positive or neutral.
The Issue with Using Separate Sigmoid Activations:
If we used sigmoid activations for each polarity (Positive, Negative, Neutral), we might get:
Positive = 0.9
Negative = 0.8
Neutral = 0.7
This does not make sense because sentiment polarities should be mutually exclusive—the sum of probabilities should be 1.
Softmax ensures this constraint by making sure all polarities sum to 1, enforcing a competition among them.
Why Concatenate Aspect#Category into One Dense Layer and Apply Binary Cross-Entropy?
1. Capturing Relationships Between Sentiments
Instead of treating each Aspect#Category,Polarity separately, we concatenate them into a single dense layer.
This helps the model learn shared patterns between aspect categories.
Example:
If HOTEL#CLEANLINESS is predicted as Positive, then it is more likely that HOTEL#QUALITY will also be Positive.
This relationship wouldn't be naturally captured if each sentiment prediction was treated independently.
2. Why Does the Model Learn These Relationships?
Even though the model is just performing matrix multiplications and applying activation functions, it can still learn relationships through weight adjustments in training.
Concatenation allows weight sharing, which enables the model to capture implicit dependencies among different sentiment polarities.
3. Binary Cross-Entropy on Each Output
The softmax constraint ensures that within each Aspect#Category, the sum of probabilities remains 1.
However, since each aspect category is independent, the binary cross-entropy loss is applied independently to each aspect.
This allows the model to optimize each Aspect#Category separately, while still ensuring mutual exclusivity among the four sentiment polarities (Positive, Negative, Neutral, None).
Summary
Softmax ensures that sentiment polarities are mutually exclusive within an Aspect#Category.
Concatenation into a shared dense layer allows the model to learn relationships between aspects (e.g., HOTEL#CLEANLINESS and HOTEL#QUALITY).
Binary Cross-Entropy is used independently on each aspect, preserving the independent multi-label nature of the task.
The network learns relationships implicitly through weight sharing and backpropagation.

