### <b> Receptive Field calculation - GoogLeNet Inception Architecture </b> <br>
==============================================================================<br>
Given below are short names used in Receptive Field computation. <br>
k - kernel size <br>
p - padding <br>
s - stride <br>
RF - Receptive Field <br>
For receptivefield calculation in Inception Module, highest RF path is considered.<br>
In inception module,  there are many parallel paths a, b, c, d connecting from previous layer to next layer in architecture,
In this case, for RF computation 1x1 followed by 5x5 path is considered, since this RF is max. 
a) 1x1
b) 1x1 and 3x3
c) 1x1 and 5x5
d) MP(3x3) and 1x1

--------------------------------------------------------------------------
| Type  		    | Input      | k,p,s    |  Output    |  RF  | Jin   |Jout |
|--------------:|-----------:|----------|:----------:| ----:| -----:|----:|
| convolution   | 224x224x3  | 7,2,2    | 112x112x64 |	7   |  1    | 2   |            
| max pool	  	| 112x112x64 | 3,1,2	  | 56x56x64	 |	11	|  2	  | 4	  |
| convolution   | 56x56x64	 | 3,0,1	| 56x56x192	 |  19  |  4	| 4   |
| max pool		| 56x56x192	 | 3,1,2    | 28x28x192	 |  27  |  4	| 8   |
| inception3a	| 28x28x192  | 5,2,1    | 28x28x256  |  59  |  8    | 8   |
| inception3b	| 28x28x256	 | 5,2,1	| 28x28x480  |  91  |  8    | 8   |
| max pool		| 28x28x480	 | 3,1,2    | 14x14x480	 |  117 |  8	| 16  |
| inception4a	| 14x14x480  | 5,2,1    | 14x14x512  |  181 |  16   | 16  |
| inception4b	| 14x14x512  | 5,2,1    | 14x14x512  |  225 |  16   | 16  |
| inception4c	| 14x14x512  | 5,2,1    | 14x14x512  |  279 |  16   | 16  |
| inception4d	| 14x14x512  | 5,2,1    | 14x14x528  |  343 |  16   | 16  |
| inception4e	| 14x14x528  | 5,2,1    | 14x14x832  |  407 |  16   | 16  |
| max pool		| 14x14x832	 | 3,1,2    | 7x7x832	 |  439 |  16	| 32  |
| inception5a	| 7x7x832    | 5,2,1    | 7x7x832    |  567 |  32   | 32  |
| inception5b	| 7x7x832    | 5,2,1    | 7x7x1024   |  595 |  32   | 32  |
| avg pool		| 7x7x1024	 | 7,0,1	| 1x1x1024	 |  947 |  32 	| 32  |	
| linear		| 1x1x1024	 | 			| 1x1x1000	 | 		|		|	  |
| softmax		| 1x1x1000	 |			| 1x1x1000	 |      |		|	  |
---------------------------------------------------------------------------	