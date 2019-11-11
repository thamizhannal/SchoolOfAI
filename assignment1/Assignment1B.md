# SchoolOfAI
<b> What are Channels and Kernels (according to EVA)? </b><br>
<b> Channels: </b><br>
Channels contains similar kind information/features bags. For example, channel for only red color, only green color, only blue color, 
only horizontal edges, only vertical edges etc. 
<b> Example: </b> In color detection, red channel detects intensity of red color that constitute that pixel.
Consider a case to where we need to detect ingredients used to make a Biriyani and an image is given. In computer vision, we use separate 
channels to detect each ingredient such as rice channel, meat channel, salt channel, garlic channel etc.

![Alt Text](https://raw.githubusercontent.com/thamizhannal/SchoolOfAI/master/images/channels.png?token=AB62OPTZNRTXX4UNYXIUDHS5ZFPPC)

<b> Kernel: </b><br>
In computer vision, kernel/filter/feature extractor is a 3x3 numerical matrix that essentially used to perform operation such as
horizontal/vertical edge detection, blurring, sharpening in the given image. This is achieved by convolving input image with kernel
that has specific numerical values to above operation specified. In Deep learning, these kernel values are randomly initialized.

![Alt Text]( https://raw.githubusercontent.com/thamizhannal/SchoolOfAI/master/images/conv3x3.gif?token=AB62OPQTE5GHHIDPHTXWT2S5ZFPSQ )

<b> Why should we only (well mostly) use 3x3 Kernels? </b>

* Kernels is matrix of odd numbers such as 3x3, 5x5, 7x7, 9x9, 11x11 etc. Since kernel <b> 3x3 is used to represent kernel of any size </b>
it is predominantly used in computer vision and deep neural networks.
* It drastically <b> reduces number of multiplications required in convolution operation </b>.
* All the GPU hardware manufactures  <b> accelerated 3x3 kernel operation in hardware level </b>. So, it is fast as compare to 5x5, 7x7. 

<b> Example: </b> consider input image of size 5x5 convolved with kernel of size 5x5 then it produces output image of size 1x1.
It requires 25 multiplication and one summation operation. Same can be achieved using 2 times of convolution operation using 3x3 kernel 
and each performs 9 multiplications and one summation totally 18 multiplications instead of 25 in 5x5 kernel. So, it reduces computation
drastically.

![Alt Text]( https://raw.githubusercontent.com/thamizhannal/SchoolOfAI/master/images/5x5_vs_2_3x3.png?token=AB62OPRVIKNAP5G67LYSHE25ZFPU6 )

<b> How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations) </b> 
We need to perform 99 times 3x3 convolution operation to reach 1x1.
199|197|195|193|191|189|187|185|183|181|179|177|175|173|171|169|167|165|163|161|
159|157|155|153|151|149|147|145|143|141|139|137|135|133|131|129|127|125|123|121|
119|117|115|113|111|109|107|105|103|101|99|97|95|93|91|89|87|85|83|81|
79|77|75|73|71|69|67|65|63|61|59|57|55|53|51|49|47|45|43|41|
39|37|35|33|31|29|27|25|23|21|19|17|15|13|11|9|7|5|3|1




