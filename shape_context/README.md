# Shape context
Scale and rotation invariant shape descriptor

Link to the article: https://medium.com/machine-learning-world/shape-context-descriptor-and-fast-characters-recognition-c031eac726f9

**What could go wrong?**

1. One of the biggest problems of all descriptors is to choosing right key-points from image. Here we used not the best algorithm of choosing them, and in a more complex example it would fail.

2. Another thing to consider is a characters vocabluary. Here for font characters we dont need it, but if you will want to make hand written symbol recognition then you will need it, cause same symbol can be written with big difference, and only one base exmaple will fail your recognition task.

**Papers to read**

1. Main doc about shape context https://github.com/creotiv/Python-Shape-Context/blob/master/info/10.1.1.112.2716.pdf
2. Good slides for understanding mechanics of algorithm https://github.com/creotiv/Python-Shape-Context/blob/master/info/ShapeContexts425.pdf

Link to Google Colab code runner to see this code in work: https://drive.google.com/file/d/16OtL71G0CNk0dSIJU2R4HrLnmj53Xfyi/view?usp=sharing

### Support
If you like my articles, you can always support me with some beer-money https://paypal.me/creotiv
