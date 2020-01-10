#include <torch/torch.h>
#include <vector>
                    
c10::IntList same_padding(torch::Tensor x, torch::Tensor k, torch::Tensor s){
    int ih = x.size(2);
    int iw = x.size(3);
    int ph;
    int pw;
    if (ih % s[0].item<int>() == 0){
        ph = std::max(k[0].item<int>() - s[0].item<int>(), 0);
    }
    else{
        ph = std::max(k[0].item<int>() - (ih % s[0].item<int>()), 0);
    }
    if (iw % s[1].item<int>() == 0){
        pw = std::max(k[1].item<int>() - s[1].item<int>(), 0);
    }
    else{
        pw = std::max(k[1].item<int>() - (iw % s[1].item<int>()), 0);
    }
    int pl = pw / 2;
    int pr = pw - pl;
    int pt = ph / 2;
    int pb = ph - pt;
    c10::IntList data({pl, pr, pt, pb}); 
    return data;
}

torch::Tensor max_pool(torch::Tensor x, int dimension, torch::Tensor k, torch::Tensor s, bool same, c10::IntList padding){
    
    if(same){
        padding = same_padding(x, k, s);
    }
    
    auto p = at::reflection_pad2d(x, padding);
    auto unfolded = p.unfold(dimension, k[0].item<int>(), s[0].item<int>()).unfold(3, k[1].item<int>(), s[1].item<int>());
    auto v = unfolded.contiguous().view({unfolded.size(0), unfolded.size(1), unfolded.size(2), unfolded.size(3),-1});
    auto pool = torch::max(v, -1);
    return std::get<0>(pool);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_pool", &max_pool, "CPPPool max_pool");
}