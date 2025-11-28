# Slice3d

3D slice visualization with VTK.

## Installation

```bash
pip install vtk-slice3d
```

## Example

```python
import numpy as np
from slice3d import Slice3d

def make_demo_data(nx, ny, nz):
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    data = (np.sin(3*np.pi*X) * np.cos(3*np.pi*Y) +
            np.sin(3*np.pi*Z) +
            np.exp(-(X**2 + Y**2 + Z**2)*4))
    return data.astype(np.float32)

nx, ny, nz = 30, 40, 50
data = make_demo_data(nx, ny, nz)

s = Slice3d(data,win_size=(300,300))
s.show(save_fn='demo.jpg')
```

# Demo Result

![demo.jpg](docs/demo.jpg)

# Supported Output Formats

* 位图(PNG、JPG): 默认 DPI = 96，无法修改
* 矢量图(PDF、SVG、EPS): 推荐使用 PDF 或 SVG 保存结果
