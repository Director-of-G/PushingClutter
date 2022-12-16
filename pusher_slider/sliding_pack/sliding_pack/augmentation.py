## Author: Yongpeng Jiang
## Contact: jyp19@mails.tsinghua.edu.cn
## Date: 16/12/2022
## -------------------------------------------------------------------
## Description:
## 
## classes for multi-slider pushing cases
## -------------------------------------------------------------------

#  import libraries
#  -------------------------------------------------------------------
import numpy as np
#  -------------------------------------------------------------------

def augment_state(x:list, beta:list, phi_r:float):
    x.append(x[0]+beta[0])
    x.append(x[1]+beta[0]*np.tan(phi_r))
    x.append(x[2])
    x.append(phi_r)

    return x
