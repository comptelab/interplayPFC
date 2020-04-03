from numpy import *

flip=True
type_ori = False



w1=pi/30
w2=pi/2
xxx=arange(0,pi-w2/2,w1)
xxx2=xxx+w2/2


xxx_f=arange(0,pi-w2/2,w1)
xxx2_f =xxx_f +w2/2 
xxx2_f = degrees(xxx2_f)

xxx_uf=arange(-pi,pi,w1)
xxx2_uf =xxx_uf +w2/2 
xxx2_uf = degrees(xxx2_uf)


if flip:
	xxx=xxx_f

else:
	xxx=xxx_uf


xxx2 =xxx +w2/2
xxx2 = degrees(xxx2)
