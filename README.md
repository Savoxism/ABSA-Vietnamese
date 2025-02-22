# ABSA-Vietnamese
This is a complete reimplementation on how to solve Aspect-based Sentiment Analysis in the Vietnamese hotel reviews.

You can read my blog to see how I approached this task:
[Aspect-based Sentiment Analysis Vietnamese](https://neuroxism.com/Sentiment+Analysis/Aspect-Based+Sentiment+Analysis+Vietnamese#1.+Setup)

![Multi-task Approach](/attachments/ACSA-v1.png)

# Dataset
- The VLSP 2018 Aspect-based Sentiment Analysis dataset:

|   Domain   |  Dataset | Reviews | Aspects | AvgLength | VocabSize | DiffVocab |
|:----------:|:--------:|:-------:|:-------:|:---------:|:---------:|:---------:|
|            | Training |  2,961  |  9,034  |     54    |   5,168   |     -     |
| Restaurant |    Dev   |  1,290  |  3,408  |     50    |   3,398   |   1,702   |
|            |   Test   |   500   |  2,419  |    163    |   3,375   |   1,729   |
|            | Training |  3,000  |  13,948 |     47    |   3,908   |     -     |
|    Hotel   |    Dev   |  2,000  |  7,111  |     23    |   2,745   |   1,059   |
|            |   Test   |   600   |  2,584  |     30    |   1,631   |    346    |

- Preprocessing: 
```mermaid 
flowchart TB
  A[Remove HTML] --> B[Unicode Norm]
  B --> C[Acronym Norm]
  C --> D[Word Segmentation]
  D --> E[Clean Text]
```

## Results
<table>
<thead>
  <tr>
    <th rowspan="2">Task</th>
    <th rowspan="2">Method</th>
    <th colspan="3">Hotel</th>
    <th colspan="3">Restaurant</th>
  </tr>
  <tr>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="5">Aspect<br>Detection</td>
    <td align="center">VLSP best submission</td>
    <td align="center">76.00</td>
    <td align="center">66.00</td>
    <td align="center">70.00</td>
    <td align="center">79.00</td>
    <td align="center">76.00</td>
    <td align="center">77.00</td>
  </tr>
  <tr>
    <td align="center">Bi-LSTM+CNN</td>
    <td align="center">84.03</td>
    <td align="center">72.52</td>
    <td align="center">77.85</td>
    <td align="center">82.02</td>
    <td align="center">77.51</td>
    <td align="center">79.70</td>
  </tr>
  <tr>
    <td align="center">BERT-based Hierarchical</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">82.06</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center"><b>84.23</b></td>
  </tr>
  <tr>
    <td align="center">Multi-task</td>
    <td align="center"><b>87.45</b></td>
    <td align="center"><b>78.17</b></td>
    <td align="center"><b>82.55</b></td>
    <td align="center">81.09</td>
    <td align="center">85.61</td>
    <td align="center">83.29</td>
  </tr>
  <tr>
    <td align="center">Multi-task Multi-branch</td>
    <td align="center">63.21</td>
    <td align="center">57.86</td>
    <td align="center">60.42</td>
    <td align="center">80.81</td>
    <td align="center">87.39</td>
    <td align="center">83.97</td>
  </tr>
  <tr>
    <td align="center" rowspan="5">Aspect +<br>Polarity</td>
    <td align="center">VLSP best submission</td>
    <td align="center">66.00</td>
    <td align="center">57.00</td>
    <td align="center">61.00</td>
    <td align="center">62.00</td>
    <td align="center">60.00</td>
    <td align="center">61.00</td>
  </tr>
  <tr>
    <td align="center">Bi-LSTM+CNN</td>
    <td align="center">76.53</td>
    <td align="center">66.04</td>
    <td align="center">70.90</td>
    <td align="center">66.66</td>
    <td align="center">63.00</td>
    <td align="center">64.78</td>
  </tr>
  <tr>
    <td align="center">BERT-based Hierarchical</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">74.69</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">71.30</td>
  </tr>
  <tr>
    <td align="center">Multi-task</td>
    <td align="center"><b>81.90</b></td>
    <td align="center"><b>73.22</b></td>
    <td align="center"><b>77.32</b></td>
    <td align="center"><b>69.66</b></td>
    <td align="center"><b>73.54</b></td>
    <td align="center"><b>71.55</b></td>
  </tr>
  <tr>
    <td align="center">Multi-task Multi-branch</td>
    <td align="center">57.55</td>
    <td align="center">52.67</td>
    <td align="center">55.00</td>
    <td align="center">68.69</td>
    <td align="center">74.29</td>
    <td align="center">71.38</td>
  </tr>
</tbody>
</table>


