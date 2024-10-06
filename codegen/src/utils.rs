use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    Ident, Type, LitInt, LitStr, Result, Token,
};
extern crate lazy_static;
use lazy_static::lazy_static;

pub enum ElementType {
    Void,
    Type(syn::Type),
}

impl quote::ToTokens for ElementType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            ElementType::Void => {
                let void_ident = quote! { () };
                void_ident.to_tokens(tokens)
            },
            ElementType::Type(ty) => ty.to_tokens(tokens),
        }
    }
}

impl Clone for ElementType {
    fn clone(&self) -> Self {
        match self {
            ElementType::Void => ElementType::Void,
            ElementType::Type(ty) => ElementType::Type(ty.clone()),
        }
    }
}

impl ElementType {
    pub fn get_bytes(&self) -> usize {
        match self {
            ElementType::Void => 0,
            ElementType::Type(_) => std::mem::size_of::<Type>(),
        }
    }
}

/// - "type", - "*mut type"
/// the former is input to native function,
/// the latter is output from native function
#[derive(PartialEq, Eq)]
pub enum ElementMode {
    Input,
    Output,
}

pub struct Element {
    pub name: Ident,
    pub ty: ElementType,
    pub mode: ElementMode,
}

pub struct ExeParser {
    pub func: Ident,
    pub result: Element,
    pub params: Vec<Element>,
}

impl Parse for ExeParser {
    fn parse(input: ParseStream) -> Result<Self> {
        let func = input.parse::<LitStr>().expect("Expected second argument to be a string literal for function name").value();
        let func = format_ident!("{}", func);

        let _comma: Option<Token![,]> = input.parse().ok();
        let result_ty = input.parse::<LitStr>().expect("Expected valid type as a string literal").value();
        let result_ty = match result_ty.as_str() {
            "" => ElementType::Void,
            _ => ElementType::Type(syn::parse_str::<Type>(&result_ty).expect("Expected valid type for result")),
        };
        let result = Element {
            name: format_ident!("result"),
            ty: result_ty,
            mode: ElementMode::Output,
        };

        let mut params = Vec::new();
        let mut i: usize = 0;
        while !input.is_empty() {
            let _comma: Option<Token![,]> = input.parse().ok();
            let mut ty_str = input.parse::<LitStr>().expect("Expected valid type as a string literal").value();
            let mode = if ty_str.starts_with("*mut ") {
                ty_str = ty_str.replace("*mut ", "");
                ElementMode::Output
            } else {
                ElementMode::Input
            };
            let ty = ElementType::Type(syn::parse_str::<Type>(&ty_str).expect("Expected valid type"));
            params.push(Element {
                name: format_ident!("param{}", i + 1),
                ty,
                mode,
            });
            i += 1;
        }

        Ok(ExeParser {
            func,
            result,
            params,
        })
    }
}

pub struct HijackParser {
    pub proc_id: LitInt,
    pub func: Ident,
    pub result: Element,
    pub params: Vec<Element>,
}

impl Parse for HijackParser {
    fn parse(input: ParseStream) -> Result<Self> {
        let proc_id = input.parse::<LitInt>().expect("Expected valid proc_id");

        let _comma: Option<Token![,]> = input.parse().ok();
        let func = input.parse::<LitStr>().expect("Expected second argument to be a string literal for function name").value();
        let func = format_ident!("{}", func);

        let _comma: Option<Token![,]> = input.parse().ok();
        let result_ty = input.parse::<LitStr>().expect("Expected valid type as a string literal").value();
        let result_ty = match result_ty.as_str() {
            "" => ElementType::Void,
            _ => ElementType::Type(syn::parse_str::<Type>(&result_ty).expect("Expected valid type for result")),
        };
        let result = Element {
            name: format_ident!("result"),
            ty: result_ty,
            mode: ElementMode::Output,
        };

        let mut params = Vec::new();
        let mut i: usize = 0;
        while !input.is_empty() {
            let _comma: Option<Token![,]> = input.parse().ok();
            let mut ty_str = input.parse::<LitStr>().expect("Expected valid type as a string literal").value();
            let mode = if ty_str.starts_with("*mut ") {
                ty_str = ty_str.replace("*mut ", "");
                ElementMode::Output
            } else {
                ElementMode::Input
            };
            let ty = ElementType::Type(syn::parse_str::<Type>(&ty_str).expect("Expected valid type"));
            params.push(Element {
                name: format_ident!("param{}", i + 1),
                ty,
                mode,
            });
            i += 1;
        }

        Ok(HijackParser {
            proc_id,
            func,
            result,
            params,
        })
    }
}

pub struct UnimplementParser {
    pub func: Ident,
    pub result: Element,
    pub params: Vec<Element>,
}

impl Parse for UnimplementParser {
    fn parse(input: ParseStream) -> Result<Self> {
        let func = input.parse::<LitStr>().expect("Expected second argument to be a string literal for function name").value();
        let func = format_ident!("{}", func);

        let _comma: Option<Token![,]> = input.parse().ok();
        let result_ty = input.parse::<LitStr>().expect("Expected valid type as a string literal").value();
        let result_ty = match result_ty.as_str() {
            "" => ElementType::Void,
            _ => ElementType::Type(syn::parse_str::<Type>(&result_ty).expect("Expected valid type for result")),
        };
        let result = Element {
            name: format_ident!("result"),
            ty: result_ty,
            mode: ElementMode::Output,
        };

        let mut params = Vec::new();
        let mut i: usize = 0;
        while !input.is_empty() {
            let _comma: Option<Token![,]> = input.parse().ok();
            let mut ty_str = input.parse::<LitStr>().expect("Expected valid type as a string literal").value();
            let mode = if ty_str.starts_with("*mut ") {
                ty_str = ty_str.replace("*mut ", "");
                ElementMode::Output
            } else {
                ElementMode::Input
            };
            let ty = ElementType::Type(syn::parse_str::<Type>(&ty_str).expect("Expected valid type"));
            params.push(Element {
                name: format_ident!("param{}", i + 1),
                ty,
                mode,
            });
            i += 1;
        }

        Ok(UnimplementParser {
            func,
            result,
            params,
        })
    }
}

lazy_static! {
    pub static ref SHADOW_DESC_TYPES: Vec<String> = {
        vec![
            "cudnnTensorDescriptor_t".to_string(),
            "cudnnFilterDescriptor_t".to_string(),
            "cudnnConvolutionDescriptor_t".to_string(),
        ]
    };

    pub static ref API_INDEX: std::collections::HashMap<&'static str, u64> = {
        let mut map = std::collections::HashMap::new();
        map.insert("cudaGetDevice", 0);
        map.insert("cudaSetDevice", 1);
        map.insert("cudaGetDeviceCount", 2);
        map.insert("cudaGetLastError", 3);
        map.insert("cudaPeekAtLastError", 4);
        map.insert("cudaStreamSynchronize", 5);
        map.insert("cudaMalloc", 6);
        map.insert("cudaMemcpy", 7);
        map.insert("cudaFree", 8);
        map.insert("cudaStreamIsCapturing", 9);
        map.insert("cudaGetDeviceProperties", 10);
        map.insert("cudaMallocManaged", 11);
        map.insert("cudaPointerGetAttributes", 12);
        map.insert("cudaHostAlloc", 13);
        map.insert("cudaFuncGetAttributes", 14);
        map.insert("cudaDeviceGetStreamPriorityRange", 15);
        map.insert("cudaMemsetAsync", 16);
        map.insert("cudaGetErrorString", 17);
        map.insert("cudaMemGetInfo", 18);
        map.insert("__cudaRegisterFatBinary", 100);
        map.insert("__cudaUnregisterFatBinary", 101);
        map.insert("__cudaRegisterFunction", 102);
        map.insert("__cudaRegisterVar", 103);
        map.insert("cudaLaunchKernel", 200);
        map.insert("cuDevicePrimaryCtxGetState", 300);
        map.insert("cuGetProcAddress", 500);
        map.insert("cuDriverGetVersion", 501);
        map.insert("cuInit", 502);
        map.insert("cuGetExportTable", 503);
        map.insert("nvmlInit_v2", 1000);
        map.insert("nvmlDeviceGetCount_v2", 1001);
        map.insert("nvmlInitWithFlags", 1002);
        map.insert("cudnnCreate", 1500);
        map.insert("cudnnCreateTensorDescriptor", 1501);
        map.insert("cudnnSetTensor4dDescriptor", 1502);
        map.insert("cudnnCreateActivationDescriptor", 1503);
        map.insert("cudnnSetActivationDescriptor", 1504);
        map.insert("cudnnActivationForward", 1505);
        map.insert("cudnnDestroy", 1506);
        map.insert("cudnnSetConvolution2dDescriptor", 1507);
        map.insert("cudnnSetStream", 1508);
        map.insert("cudnnSetTensorNdDescriptor", 1509);
        map.insert("cudnnDestroyTensorDescriptor", 1510);
        map.insert("cudnnCreateFilterDescriptor", 1511);
        map.insert("cudnnDestroyFilterDescriptor", 1512);
        map.insert("cudnnSetFilterNdDescriptor", 1513);
        map.insert("cudnnCreateConvolutionDescriptor", 1514);
        map.insert("cudnnDestroyConvolutionDescriptor", 1515);
        map.insert("cudnnSetConvolutionNdDescriptor", 1516);
        map.insert("cudnnSetConvolutionGroupCount", 1517);
        map.insert("cudnnSetConvolutionMathType", 1518);
        map.insert("cudnnSetConvolutionReorderType", 1519);
        map.insert("cudnnGetConvolutionForwardAlgorithm_v7", 1520);
        map.insert("cudnnConvolutionForward", 1521);
        map.insert("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize", 1522);
        map.insert("cudnnGetBatchNormalizationTrainingExReserveSpaceSize", 1523);
        map.insert("cudnnBatchNormalizationForwardTrainingEx", 1524);
        map.insert("cudnnGetBatchNormalizationBackwardExWorkspaceSize", 1525);
        map.insert("cudnnBatchNormalizationBackwardEx", 1526);
        map.insert("cudnnGetConvolutionBackwardDataAlgorithm_v7", 1527);
        map.insert("cudnnConvolutionBackwardData", 1528);
        map.insert("cudnnGetConvolutionBackwardFilterAlgorithm_v7", 1529);
        map.insert("cudnnConvolutionBackwardFilter", 1530);
        map.insert("cudnnBatchNormalizationForwardInference", 1531);
        map.insert("cudnnSetFilter4dDescriptor", 1532);
        map.insert("cudnnGetConvolutionNdForwardOutputDim", 1533);
        map.insert("cudnnGetConvolutionForwardWorkspaceSize", 1534);
        map.insert("cudnnGetErrorString", 1535);
        map.insert("cublasCreate_v2", 2000);
        map.insert("cublasDestroy_v2", 2001);
        map.insert("cublasSetStream_v2", 2002);
        map.insert("cublasSetMathMode", 2003);
        map.insert("cublasSgemm_v2", 2004);
        map.insert("cublasSgemmStridedBatched", 2005);
        map.insert("cublasGetMathMode", 2006);
        map.insert("cublasGemmEx", 2007);
        map.insert("cublasGemmStridedBatchedEx", 2008);
        map
    };
}

pub fn get_success_status(ty: &str) -> &str {
    match ty {
        "cublasStatus_t" => "CUBLAS_STATUS_SUCCESS",
        "CUresult" => "CUDA_SUCCESS",
        "cudaError_t" => "cudaSuccess",
        "cudnnStatus_t" => "CUDNN_STATUS_SUCCESS",
        "nvmlReturn_t" => "NVML_SUCCESS",
        &_ => todo!(),
    }
}

pub fn get_api_index(api: &str) -> u64 {
    match API_INDEX.get(api) {
        Some(id) => *id,
        _ => panic!("invalid API {}", api),
    }
}