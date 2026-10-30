import torch
import datasets
import transformers

def test_cuda_and_libraries():
    assert torch.cuda.is_available(), "CUDA must be available"
    device_name = torch.cuda.get_device_name(0)
    assert "RTX" in device_name or "GeForce" in device_name, f"Expected RTX GPU, got {device_name}"
    print(f"CUDA verified: {device_name}, Torch: {torch.__version__}, Transformers: {transformers.__version__}, Datasets: {datasets.__version__}")

if __name__ == "__main__":
    test_cuda_and_libraries()
