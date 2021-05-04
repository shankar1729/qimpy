#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cstdio>

namespace py = pybind11;

//Add format string bindings for Torch's complex types
namespace pybind11
{	template<> struct format_descriptor<c10::complex<float>> { static std::string format() { return std::string("Zf"); } };
	template<> struct format_descriptor<c10::complex<double>> { static std::string format() { return std::string("Zd"); } };
}

struct BufferView
{
	//Buffer contents:
	void* data_ptr; //underlying data pointer
	py::ssize_t itemsize; //number of bytes per atom
	std::string format; //format string
	std::vector<py::ssize_t> shape; //dimensions in items
	std::vector<py::ssize_t> strides; //strides in bytes
	
	//Bind a view to a specified tensor:
	BufferView(torch::Tensor t)
	{	data_ptr = t.data_ptr();
		AT_DISPATCH_ALL_TYPES_AND_COMPLEX(t.scalar_type(), "BufferView",
		([&]{
				itemsize = sizeof(scalar_t);
				format = py::format_descriptor<scalar_t>::format();
			} 
		));
		for(auto size: t.sizes()) shape.push_back(size);
		for(auto stride: t.strides()) strides.push_back(stride * itemsize);
	}
	
	//Expose a buffer interface to underlying tensor:
	py::buffer_info getBuffer()
	{	return py::buffer_info(data_ptr, itemsize, format, shape.size(), shape, strides);
	}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	py::class_<BufferView>(m, "BufferView", py::buffer_protocol())
		.def(py::init<torch::Tensor>())
		.def_buffer(&BufferView::getBuffer);
}
