from typing import List
from vtkmodules.util import numpy_support
import matplotlib
import numpy as np
import quaternion
import vtk


def _place(axis: int, a: float, pos0: float, pos1: float):
	"""Return [x,y,z] where value a is placed at `axis`,
	and the remaining axes (in ascending order) get pos0, pos1."""
	v = [None, None, None]
	v[axis] = a
	rem = [i for i in (0, 1, 2) if i != axis]
	v[rem[0]] = pos0
	v[rem[1]] = pos1
	return v


class Viewer:
	def __init__(self, data, size=(2, 2, 2), cmap='jet', q0=(1, 0, 0, 0), cam_pos=(5, -5, -5), cam_focal=(0, 0, 0),
	             cam_up=(1, 0, 0), cam_clip=(3, 15), bg=(1., 1., 1.), show_orientation_marker=True, show_outline=True,
	             win_name='slice3d', projection=True, win_size=(600, 600)):
		data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
		self.nx, self.ny, self.nz = data.shape
		self.size = size
		self.vtk_data = vtk.vtkImageData()
		self.vtk_data.SetOrigin(-size[0] / 2, -size[1] / 2, -size[2] / 2)
		self.vtk_data.SetExtent(0, self.nx - 1, 0, self.ny - 1, 0, self.nz - 1)
		self.vtk_data.SetSpacing(size[0] / (self.nx - 1), size[1] / (self.ny - 1), size[2] / (self.nz - 1))
		flat_data = data.ravel(order='F')
		vtk_array = numpy_support.numpy_to_vtk(num_array=flat_data, deep=True, array_type=vtk.VTK_FLOAT)
		self.vtk_data.GetPointData().SetScalars(vtk_array)

		self.q0 = quaternion.quaternion(*q0)
		self.q = quaternion.quaternion(*q0)
		self.rot_axis = {'x': np.array([1, 0, 0]), 'y': np.array([0, 1, 0]), 'z': np.array([0, 0, 1])}
		self.bounds = (-size[0]/2, size[0]/2, -size[1]/2, size[1]/2, -size[2]/2, size[2]/2)
		self.text_anchors = []

		self.cam_pos = cam_pos
		self.cam_focal = cam_focal
		self.cam_up = cam_up
		self.cam_clip = cam_clip

		self.lut = self.make_cmap(cmap)
		self.renderer = vtk.vtkRenderer()
		self.renderer.SetBackground(*bg)
		self.assembly = vtk.vtkAssembly()
		self.camera = self.renderer.GetActiveCamera()
		self.render_window = vtk.vtkRenderWindow()
		self.render_window.AddRenderer(self.renderer)
		self.render_window.SetWindowName(win_name)
		self.render_window.SetSize(*win_size)
		self.interactor = vtk.vtkRenderWindowInteractor()
		self.interactor.SetRenderWindow(self.render_window)
		if show_orientation_marker:
			self.add_orientation_marker()
		if show_outline:
			self.add_outline()
		if projection:
			self.camera.ParallelProjectionOn()
		else:
			self.camera.ParallelProjectionOff()

	def make_cmap(self, x):
		if isinstance(x, str):  # 情况一：传入 colormap 名称
			cmap = matplotlib.colormaps.get_cmap(x)
			colors = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)  # 转成 0-255
		else:  # 情况二：传入 numpy 数组
			arr = np.asarray(x)
			if arr.shape[0] != 256 or arr.shape[1] not in (3, 4):
				raise ValueError('数组必须是 256x3 或 256x4')
			if arr.shape[1] == 3:
				alpha = np.ones((256, 1), dtype=arr.dtype)
				arr = np.hstack([arr, alpha])
			vmin, vmax = arr.min(), arr.max()
			if vmin >= 0 and vmax <= 255:
				colors = arr.astype(np.uint8)
			else:
				arr = (arr - vmin) / (vmax - vmin)
				colors = (arr * 255).astype(np.uint8)
		lut = vtk.vtkLookupTable()
		lut.SetNumberOfTableValues(256)
		lut.Build()
		vtk_arr = vtk.vtkUnsignedCharArray()
		vtk_arr.SetNumberOfComponents(4)
		vtk_arr.SetNumberOfTuples(256)
		vtk_arr.SetVoidArray(colors, 256 * 4, 1)  # 直接绑定 numpy 内存
		lut.SetTable(vtk_arr)
		return lut

	def make_actor(self, source, color, linewidth=None, lightingoff=False):
		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(source.GetOutputPort())
		actor = vtk.vtkActor()
		actor.SetMapper(mapper)
		actor.GetProperty().SetColor(color)
		if lightingoff:
			actor.GetProperty().LightingOff()
		if linewidth is not None:
			actor.GetProperty().SetLineWidth(linewidth)
		self.assembly.AddPart(actor)

	def add_orientation_marker(self):
		self.axes_marker = vtk.vtkAxesActor()
		self.axes_marker.SetTotalLength(1.0, 1.0, 1.0)
		self.axes_marker.SetCylinderRadius(0.02)
		self.axes_marker.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 0, 0)  # X red
		self.axes_marker.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 1, 0)  # Y green
		self.axes_marker.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 1)  # Z blue
		self.marker = vtk.vtkOrientationMarkerWidget()
		self.marker.SetOrientationMarker(self.axes_marker)
		self.marker.SetInteractor(self.interactor)
		self.marker.SetViewport(0.80, 0.80, 1.00, 1.00)
		self.marker.EnabledOn()
		self.marker.InteractiveOff()

	def add_outline(self):
		cube = vtk.vtkCubeSource()
		cube.SetBounds(self.bounds)
		cube.Update()
		outline = vtk.vtkOutlineFilter()
		outline.SetInputConnection(cube.GetOutputPort())
		self.make_actor(outline, (0, 0, 0), 2)

	def reset(self):
		self.camera.SetPosition(*self.cam_pos)
		self.camera.SetFocalPoint(*self.cam_focal)
		self.camera.SetViewUp(*self.cam_up)
		self.camera.SetClippingRange(*self.cam_clip)
		self.render_window.Render()

	def show(self, save_fn=None):
		self.renderer.AddActor(self.assembly)
		self.renderer.ResetCamera()
		self.reset()
		self.render_window.Render()
		if save_fn is not None:
			ext = save_fn.lower().split('.')[-1]
			if ext in ('pdf', 'svg', 'eps', 'ps'):
				exporter = vtk.vtkGL2PSExporter()
				if ext == 'pdf':
					exporter.SetFileFormatToPDF()
				elif ext == 'svg':
					exporter.SetFileFormatToSVG()
					exporter.SetCompress(False)
				elif ext == 'eps':
					exporter.SetFileFormatToEPS()
					exporter.SetCompress(False)
				elif ext == 'ps':
					exporter.SetFileFormatToPS()
					exporter.SetCompress(False)
				exporter.SetFilePrefix(save_fn[:-len(ext) - 1])
				exporter.SetRenderWindow(self.render_window)
				exporter.Write()
			elif ext in ('png', 'jpg', 'jpeg'):
				w2i = vtk.vtkWindowToImageFilter()
				w2i.SetInput(self.render_window)
				w2i.Update()
				if ext == 'png':
					writer = vtk.vtkPNGWriter()
				else:
					writer = vtk.vtkJPEGWriter()
				writer.SetFileName(save_fn)
				writer.SetInputConnection(w2i.GetOutputPort())
				writer.Write()
			else:
				raise ValueError('不支持的文件格式: ' + save_fn)
			print(f'已保存图片到 {save_fn}')
		self.interactor.Start()


class SceneViewer(Viewer):
	def __init__(self, data, step=(15, 15, 15), **kwargs):
		super().__init__(data, **kwargs)
		self.step = {'x': step[0], 'y': step[1], 'z': step[2]}
		self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self.keypress_callback)

	def reset(self):
		self.q = self.q0
		self.apply_quaternion()
		self.update_text_positions()
		super().reset()

	def apply_quaternion(self):
		R = quaternion.as_rotation_matrix(self.q)
		M = np.eye(4)
		M[:3, :3] = R
		mat = vtk.vtkMatrix4x4()
		mat.DeepCopy(M.flatten())
		t = vtk.vtkTransform()
		t.SetMatrix(mat)
		self.assembly.SetUserTransform(t)

	def rotate_point_by_quaternion(self, p):
		v = quaternion.quaternion(0, *p)
		r = self.q * v * self.q.conjugate()
		return np.array([r.x, r.y, r.z])

	def update_text_positions(self):
		for nd, text_actor, anchor in self.text_anchors:  # [(nd, actor, (x,y,z)), ...]
			new_pos = self.rotate_point_by_quaternion(anchor)
			if nd == 2:
				coord = text_actor.GetPositionCoordinate()
				coord.SetCoordinateSystemToWorld()
				coord.SetValue(*new_pos)  # 世界坐标
			else:
				text_actor.SetPosition(new_pos)

	def make_plane(self, plane):
		cutter = vtk.vtkCutter()
		cutter.SetCutFunction(plane)
		cutter.SetInputData(self.vtk_data)
		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(cutter.GetOutputPort())
		mapper.SetLookupTable(self.lut)
		mapper.SetColorModeToMapScalars()
		mapper.SetScalarRange(0, 255)  # 数据范围
		actor = vtk.vtkActor()
		actor.SetMapper(mapper)
		actor.GetProperty().LightingOff()
		return actor

	def add_slice(self, axis: int, index: int, loc=None, c=(0, 0, 0), lw=2):
		assert axis in (0, 1, 2)
		dims = [self.nx, self.ny, self.nz]
		assert 0 <= index < dims[axis]
		origin = [0, 0, 0]
		if index == dims[axis] - 1:
			origin[axis] = self.size[axis]/2 - 1e-6
		else:
			origin[axis] = self.size[axis] * index / (dims[axis] - 1) - self.size[axis]/2
		normal = [0, 0, 0]
		normal[axis] = 1
		plane = vtk.vtkPlane()
		plane.SetOrigin(*origin)
		plane.SetNormal(*normal)
		actor = self.make_plane(plane)
		# If loc is specified, move the actor to the loc position for display and draw a border line at the original position
		if loc is not None:
			pos = self.size[axis] * loc / (dims[axis] - 1) - self.size[axis]/2 - origin[axis]
			origin[axis] = pos
			actor.SetPosition(*origin)
			cutter = vtk.vtkCutter()
			cutter.SetCutFunction(plane)
			cutter.SetInputData(self.vtk_data)
			outline = vtk.vtkOutlineFilter()
			outline.SetInputConnection(cutter.GetOutputPort())
			self.make_actor(outline, color=c, linewidth=lw)
		self.assembly.AddPart(actor)

	def add_axis_ticks(self, axis: int, parallel_axis: int, ticks: List[int], labels: List[str], title='Title', title_ha='left',title_va='bottom',tick_length=0.2,
	                   axis_pos=(1, 1), plus_dir=True, label_offset=(0, 0, 0), title_offset=(0, 0, 0), title_angle=0):
		assert axis in (0, 1, 2)
		assert parallel_axis in (0, 1, 2)
		assert parallel_axis != axis
		assert len(ticks) == len(labels)
		assert title_ha in ['left', 'center', 'right']
		assert title_va in ['top', 'center', 'bottom']
		dims = (self.nx, self.ny, self.nz)
		pos0, pos1 = axis_pos
		l0, l1, l2 = label_offset
		if not plus_dir:
			tick_length = -tick_length

		for t_idx, lab in zip(ticks, labels):
			# index -> [-1, 1]
			t = 2 * t_idx / (dims[axis] - 1) - 1

			# line endpoints
			p1 = _place(axis, t, pos0, pos1)
			p2 = p1.copy()
			p2[parallel_axis] += tick_length

			# text position (with offsets applied component-wise)
			text_pos = [p1[0] + l0, p1[1] + l1, p1[2] + l2]

			line = vtk.vtkLineSource()
			line.SetPoint1(*p1)
			line.SetPoint2(*p2)
			self.make_actor(line, (0, 0, 0))

			text = vtk.vtkBillboardTextActor3D()
			text.SetInput(lab)
			text.SetPosition(*text_pos)
			text.GetTextProperty().SetFontSize(12)
			text.GetTextProperty().SetColor(0, 0, 0)
			self.text_anchors.append((3, text, text_pos))
			self.renderer.AddActor(text)

		# Title
		t0, t1, t2 = title_offset
		title_pos = _place(axis, t0, pos0 + t1, pos1 + t2)

		text_actor = vtk.vtkTextActor()
		text_actor.SetInput(title)
		coord = text_actor.GetPositionCoordinate()
		coord.SetCoordinateSystemToWorld()
		coord.SetValue(*title_pos)

		self.text_anchors.append((2, text_actor, title_pos))
		prop = text_actor.GetTextProperty()
		prop.SetFontSize(18)
		prop.SetColor(0, 0, 0)
		prop.SetOrientation(title_angle)
		if title_ha=='left':
			prop.SetJustificationToLeft()
		elif title_ha=='center':
			prop.SetJustificationToCentered()
		else:
			prop.SetJustificationToRight()
		if title_va == 'bottom':
			prop.SetVerticalJustificationToBottom()
		elif title_va=='center':
			prop.SetVerticalJustificationToCentered()
		else:
			prop.SetVerticalJustificationToTop()
		self.renderer.AddActor2D(text_actor)

	def add_line(self, p1, p2, c=(0.5, 0.8, 0.2), lw=2., lightingoff=False):
		line = vtk.vtkLineSource()
		line.SetPoint1(*p1)
		line.SetPoint2(*p2)
		self.make_actor(line, c, lw, lightingoff)

	def add_cone(self, tip=(0, 0, 0), direction=(1, 0, 0), height=0.3, radius=0.1, c=(0, 1, 0), res=20, lightingoff=False):
		"""
		tip: 圆锥顶点坐标
		direction: 圆锥轴方向
		height: 圆锥高度
		radius: 底面半径
		c: RGB颜色
		res: 底面圆周的分段数
		"""
		direction = np.array(direction, dtype=np.float32)
		direction = direction / np.linalg.norm(direction)
		center = np.array(tip) - 0.5 * height * direction
		cone = vtk.vtkConeSource()
		cone.SetCenter(*center)
		cone.SetDirection(*direction)
		cone.SetHeight(height)
		cone.SetRadius(radius)
		cone.SetResolution(res)
		self.make_actor(cone, c, lightingoff)

	def add_arrow(self, p1, p2, c=(1., 0., 0.), lw=None, cone_h=None, cone_r=None, cone_res=None, lightingoff=False):
		v = np.array(p2) - np.array(p1)
		norm = np.linalg.norm(v)
		if norm == 0:
			return
		direction = v / norm
		if lw is None:
			lw = max(1., norm * 0.05)  # 线宽随长度调整
		if cone_h is None:
			cone_h = norm * 0.2  # 箭头高度 = 线段长度的 20%
		if cone_r is None:
			cone_r = cone_h * 0.4  # 箭头底面半径 = 高度的 40%
		if cone_res is None:
			cone_res = 20
		self.add_line(p1, p2, c=c, lw=lw, lightingoff=lightingoff)
		self.add_cone(p2, direction, height=cone_h, radius=cone_r, c=c, res=cone_res, lightingoff=lightingoff)

	def add_sphere(self, center=(0, 0, 0), radius=0.2, c=(1, 0, 0), res=20, lightingoff=False):
		"""
		center: 球心坐标
		radius: 半径
		c: RGB颜色
		res: 球面分辨率
		"""
		sphere = vtk.vtkSphereSource()
		sphere.SetCenter(*center)
		sphere.SetRadius(radius)
		sphere.SetThetaResolution(res)  # 经度方向分段数
		sphere.SetPhiResolution(res)  # 纬度方向分段数
		self.make_actor(sphere, c, lightingoff)

	def add_cube(self, center=(0, 0, 0), size=(0.2, 0.2, 0.2), c=(0.8, 0.5, 0.2), lightingoff=False):
		"""
		center: 长方体中心坐标
		size: 边长
		c: RGB颜色
		"""
		cube = vtk.vtkCubeSource()
		cube.SetCenter(*center)
		cube.SetXLength(size[0])
		cube.SetYLength(size[1])
		cube.SetZLength(size[2])
		self.make_actor(cube, c, lightingoff)

	def add_cylinder(self, center=(0, 0, 0), direction=(0, 1, 0),
	                 height=0.5, radius=0.2, c=(0.2, 0.7, 0.4), res=20, lightingoff=False):
		"""
		center: 圆柱中心坐标
		direction: 圆柱轴方向
		height: 高度
		radius: 半径
		c: RGB颜色
		res: 圆周分段数
		"""
		direction = np.array(direction, dtype=np.float32)
		direction = direction / np.linalg.norm(direction)
		cylinder = vtk.vtkCylinderSource()
		cylinder.SetCenter(*center)
		cylinder.SetHeight(height)
		cylinder.SetRadius(radius)
		cylinder.SetResolution(res)
		transform = vtk.vtkTransform()
		z_axis = np.array([0, 0, 1])
		axis = np.cross(z_axis, direction)
		angle = np.degrees(np.arccos(np.dot(z_axis, direction)))
		if np.linalg.norm(axis) > 1e-6:
			transform.RotateWXYZ(angle, *axis)
		transform_filter = vtk.vtkTransformPolyDataFilter()
		transform_filter.SetTransform(transform)
		transform_filter.SetInputConnection(cylinder.GetOutputPort())
		self.make_actor(transform_filter, c, lightingoff)

	def keypress_callback(self, obj, event):
		key = obj.GetKeySym()
		ctrl = obj.GetControlKey()
		if key == 'i':  # 按下 i 键
			print('Camera position:', self.camera.GetPosition())
			print('Camera focal point:', self.camera.GetFocalPoint())
			print('Camera view up:', self.camera.GetViewUp())
			print('Camera clipping range:', self.camera.GetClippingRange())
			print(self.q)
		elif key == 'r':  # 恢复初始状态
			self.reset()
		elif key in ('x', 'y', 'z'):
			sign = -1 if ctrl else 1
			dq = quaternion.from_rotation_vector(self.rot_axis[key] * np.radians(sign * self.step[key]))
			self.q = self.q * dq
			self.apply_quaternion()
			self.update_text_positions()
			self.render_window.Render()


class Slice3d(Viewer):
	def __init__(self, data, xlabel='X', ylabel='Y', zlabel='Z', xlabel_pos=((0.1, 0.9), (0.3, 0.9)),
	             ylabel_pos=((0.1, 0.8), (0.3, 0.8)), zlabel_pos=((0.1, 0.7), (0.3, 0.7)), **kwargs):
		super().__init__(data, **kwargs)
		xsize, ysize, zsize = self.size[0]/2, self.size[1] / 2, self.size[2] / 2
		self.planeX = vtk.vtkImagePlaneWidget()
		self.planeX.SetInteractor(self.interactor)
		self.planeX.SetInputData(self.vtk_data)
		self.planeX.SetPlaneOrientationToXAxes()
		self.planeX.SetOrigin(-xsize, -ysize, -zsize)
		self.planeX.SetPoint1(-xsize,  ysize, -zsize)
		self.planeX.SetPoint2(-xsize, -ysize,  zsize)
		self.planeX.DisplayTextOn()
		self.planeX.SetSliceIndex(0)
		self.planeX.On()

		self.planeY = vtk.vtkImagePlaneWidget()
		self.planeY.SetInteractor(self.interactor)
		self.planeY.SetInputData(self.vtk_data)
		self.planeY.SetPlaneOrientationToYAxes()
		self.planeY.SetOrigin(-xsize, ysize, -zsize)
		self.planeY.SetPoint1(-xsize, ysize,  zsize)
		self.planeY.SetPoint2( xsize, ysize, -zsize)
		self.planeY.DisplayTextOn()
		self.planeY.SetSliceIndex(self.ny - 1)
		self.planeY.On()

		self.planeZ = vtk.vtkImagePlaneWidget()
		self.planeZ.SetInteractor(self.interactor)
		self.planeZ.SetInputData(self.vtk_data)
		self.planeZ.SetPlaneOrientationToZAxes()
		self.planeZ.SetOrigin(-xsize, -ysize, zsize)
		self.planeZ.SetPoint1( xsize, -ysize, zsize)
		self.planeZ.SetPoint2(-xsize,  ysize, zsize)
		self.planeZ.DisplayTextOn()
		self.planeZ.SetSliceIndex(self.nz - 1)
		self.planeZ.On()

		self.planeX.SetLookupTable(self.lut)
		self.planeY.SetLookupTable(self.lut)
		self.planeZ.SetLookupTable(self.lut)
		self.sliderX = self.make_slider(xlabel, (0, self.nx - 1), 0, xlabel_pos, self.update_x)
		self.sliderY = self.make_slider(ylabel, (0, self.ny - 1), self.ny - 1, ylabel_pos, self.update_y)
		self.sliderZ = self.make_slider(zlabel, (0, self.nz - 1), self.nz - 1, zlabel_pos, self.update_z)

	def make_slider(self, label, rng, value, pos, callback):
		rep = vtk.vtkSliderRepresentation2D()
		rep.SetMinimumValue(rng[0])
		rep.SetMaximumValue(rng[1])
		rep.SetValue(value)
		rep.SetTitleText(label)

		rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
		rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
		rep.GetPoint1Coordinate().SetValue(*pos[0])
		rep.GetPoint2Coordinate().SetValue(*pos[1])

		rep.SetSliderLength(0.02)
		rep.SetSliderWidth(0.03)
		rep.SetEndCapLength(0.01)
		rep.SetTitleHeight(0.02)
		rep.SetLabelHeight(0.02)

		rep.GetTubeProperty().SetColor(0, 0, 0)  # 线
		rep.GetSliderProperty().SetColor(1, 0, 0)  # 滑块
		rep.GetCapProperty().SetColor(0, 0, 1)  # 端点
		rep.GetTitleProperty().SetColor(0, 0, 0)  # 标题
		rep.GetLabelProperty().SetColor(0, 0, 0)  # 数值标签

		slider = vtk.vtkSliderWidget()
		slider.SetInteractor(self.interactor)
		slider.SetRepresentation(rep)
		slider.AddObserver(vtk.vtkCommand.InteractionEvent, callback)
		slider.EnabledOn()
		return slider

	def update_x(self, obj, evt):
		val = round(obj.GetRepresentation().GetValue())
		self.planeX.SetSliceIndex(val)
		self.render_window.Render()

	def update_y(self, obj, evt):
		val = round(obj.GetRepresentation().GetValue())
		self.planeY.SetSliceIndex(val)
		self.render_window.Render()

	def update_z(self, obj, evt):
		val = round(obj.GetRepresentation().GetValue())
		self.planeZ.SetSliceIndex(val)
		self.render_window.Render()
