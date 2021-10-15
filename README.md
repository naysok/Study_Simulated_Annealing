# Study_Simulated_Annealing  

数式を定義すべきだが面倒なので Rhino で NxN の形の PointArray を用意し、この PointArray から Polygon を作成し計算に用いる（実際には、Polygon として計算にかけるのは近傍のみとする）。  

[images](_images/image_1.png)  

計算する近傍のポリゴンを絞るために、先に計算領域を産出し今計算する点がどの領域であるか判別し、その領域の Polygon を計算する。  

[images](_images/image_2.png)  

XY 座標を変数に取り、Polygon へ Ray を飛ばし Interset の座標を求めその Z 座標を返り値をする。  

[images](_images/image_0.png)  

返り値を評価し、小さいほうへ探索していく。  





### Result  

2変数の時の最小値（設計値）  

```
X : 13.519914  
Y : 13.317021
Z : 0.0
```


### Ref  


RayTriangleIntersection  
  [https://github.com/naysok/Mesh_Vertex_Color/blob/c6fafe480957305176ac1adc14c093d9278baa94/mesh_vertex_color/np_ray_triangle_intersection.py](https://github.com/naysok/Mesh_Vertex_Color/blob/c6fafe480957305176ac1adc14c093d9278baa94/mesh_vertex_color/np_ray_triangle_intersection.py)  #   S t u d y _ S i m u l a t e d _ A n n e a l i n g  
 