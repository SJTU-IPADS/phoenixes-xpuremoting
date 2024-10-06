extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, parse_str, Ident, Type};

mod utils;
use utils::{
    Element, ElementMode, ExeParser, HijackParser, UnimplementParser,
    get_success_status, get_api_index,
};
#[cfg(feature = "shadow_desc")]
use utils::SHADOW_DESC_TYPES;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The derive macros for auto trait impl.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The derive procedural macro to generate `Transportable` trait implementation for a *fixed-size* type.
///
/// ### Example
/// To use this macro, annotate a struct or enum with `#[derive(Transportable)]`.
///
/// ```ignore
/// #[derive(Transportable)]
/// pub struct MyStruct {
///     ...
/// }
/// ```
///
/// This invocation generates a `Transportable` trait implementation for `MyStruct`.
/// The `Transportable` trait is used for sending and receiving the struct over a communication channel.
///
/// Specifically, the implementation is expanded as:
///
/// ```ignore
/// impl Transportable for MyStruct {
///     fn send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
///         let memory = RawMemory::new(self, std::mem::size_of::<Self>());
///         match channel.put_bytes(&memory)? == std::mem::size_of::<Self>() {
///             true => Ok(()),
///             false => Err(CommChannelError::IoError),
///         }
///     }
///
///     fn recv<T: CommChannel>(&mut self, channel: &mut T) -> Result<(), CommChannelError> {
///         let mut memory = RawMemoryMut::new(self, std::mem::size_of::<Self>());
///         match channel.get_bytes(&mut memory)? == std::mem::size_of::<Self>() {
///             true => Ok(()),
///             false => Err(CommChannelError::IoError),
///         }
///     }
/// }
/// ```
///
#[proc_macro_derive(Transportable)]
pub fn transportable_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let name = &input.ident;

    let gen = quote! {
        impl Transportable for #name {
            fn emulate_send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
                let memory = RawMemory::new(self, std::mem::size_of::<Self>());
                match channel.put_bytes(&memory)? == std::mem::size_of::<Self>() {
                    true => Ok(()),
                    false => Err(CommChannelError::IoError),
                }
            }

            fn send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
                let memory = RawMemory::new(self, std::mem::size_of::<Self>());
                match channel.put_bytes(&memory)? == std::mem::size_of::<Self>() {
                    true => {
                        Ok(())},
                    false => Err(CommChannelError::IoError),
                }
            }

            fn recv<T: CommChannel>(&mut self, channel: &mut T) -> Result<(), CommChannelError> {
                let mut memory = RawMemoryMut::new(self, std::mem::size_of::<Self>());
                match channel.get_bytes(&mut memory)? == std::mem::size_of::<Self>() {
                    true => Ok(()),
                    false => Err(CommChannelError::IoError),
                }
            }
        }
    };

    gen.into()
}

#[proc_macro_derive(MemorySize)]
pub fn memorysize_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let name = &input.ident;

    let gen = quote! {
        impl MemorySize for #name {
            fn mem_size(&self) -> usize {
                std::mem::size_of::<Self>()
            }
        }
    };

    gen.into()
}

/// The derive procedural macro to defaultly zero a type variable.
///
/// ### Example
/// To use this macro, annotate a struct or enum with `#[derive(ZeroDefault)]`.
///
/// ```ignore
/// #[derive(ZeroDefault)]
/// pub struct MyStruct {
///    ...
/// }
/// ```
/// 
/// This invocation generates a `Default` trait implementation for `MyStruct`,
/// which is used for initializing the struct with zeroed memory.
/// 
/// Specifically, the implementation is expanded as:
/// 
/// ```ignore
/// impl Default for MyStruct {
///    fn default() -> Self {
///       unsafe { ::std::mem::zeroed() }
///   }
/// }
/// ```
/// 
#[proc_macro_derive(ZeroDefault)]
pub fn defaultable_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let name = &input.ident;

    let gen = quote! {
        impl Default for #name {
            fn default() -> Self {
                unsafe { ::std::mem::zeroed() }
            }
        }
    };

    gen.into()
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The collection of the procedural macro to generate Rust functions for client usage.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate hijack functions for client intercepting application calls.
///
/// To use this macro, annotate a call to `gen_hijack` with specified proc_id and the desired function name,
/// followed by the types of the return and parameters as string literals.
///
/// ### Example
/// We have a function `cudaGetDevice` with the following signature:
///
/// ```ignore
/// pub fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
/// ```
///
/// To use this macro generating a hijack function for interception, we can write:
///
/// ```ignore
/// gen_hijack!("0", "cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
/// ```
///
/// This invocation generates a function `cudaGetDevice` with the same signature as the original function,
/// which will intercept the call to `cudaGetDevice` and send the parameters to the server then wait for the result.
///
/// Specifically, the function is expanded as:
///
/// ```ignore
/// #[no_mangle]
/// pub extern "C" fn cudaGetDevice(param1: *mut ::std::os::raw::c_int) -> cudaError_t {
///     assert_eq!(true, *ENABLE_LOG);
///     info!("[{}:{}] cudaGetDevice", std::file!(), std::line!());
///     let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
///     let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
///     let proc_id = 0;
///     let mut var1: ::std::os::raw::c_int = Default::default();
///     let mut result: cudaError_t = Default::default();

///     match proc_id.send(channel_sender) {
///         Ok(()) => {}
///         Err(e) => panic!("failed to send proc_id: {:?}", e),
///     }
///     match channel_sender.flush_out() {
///         Ok(()) => {}
///         Err(e) => panic!("failed to send: {:?}", e),
///     }

///     match var1.recv(channel_receiver) {
///         Ok(()) => {}
///         Err(e) => panic!("failed to receive var1: {:?}", e),
///     }
///     match result.recv(channel_receiver) {
///         Ok(()) => {}
///         Err(e) => panic!("failed to receive result: {:?}", e),
///     }
///     unsafe {
///         *param1 = var1;
///     }
///     return result;
/// }
/// ```
///
#[proc_macro]
pub fn gen_hijack(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as HijackParser);

    let (proc_id, func, result, params) = (input.proc_id, input.func, input.result, input.params);

    let vars: Vec<Element> = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .enumerate()
        .map(|(i, param)| Element {
            name: format_ident!("var{}", i + 1),
            ty: param.ty.clone(),
            mode: ElementMode::Output,
        })
        .collect();

    // definition statements
    let def_statements = vars.iter().map(|var| {
        let name = &var.name;
        let ty = &var.ty;
        quote! { let mut #name: #ty = Default::default(); }
    });

    // send parameters
    let send_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            quote! {
                match #name.send(channel_sender) {
                    Ok(()) => {}
                    Err(e) => panic!("failed to send #name: {:?}", e),
                }
            }
        });

    // receive vars
    let recv_statements = vars.iter().map(|var| {
        let name = &var.name;
        quote! {
            match #name.recv(channel_receiver) {
                Ok(()) => {}
                Err(e) => panic!("failed to receive #name: {:?}", e),
            }
        }
    });

    // assign vars to params
    let assign_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .zip(vars.iter())
        .map(|(param, var)| {
            let param_name = &param.name;
            let var_name = &var.name;
            quote! { unsafe { *#param_name = #var_name; } }
        });

    let params = params.iter().map(|param| {
        let name = &param.name;
        let ty = &param.ty;
        match param.mode {
            ElementMode::Input => quote! { #name: #ty },
            ElementMode::Output => quote! { #name: *mut #ty },
        }
    });
    let result_name = &result.name;
    let result_ty = &result.ty;
    let gen_fn = quote! {
        #[no_mangle]
        pub extern "C" fn #func(#(#params),*) -> #result_ty {
            assert_eq!(true, *ENABLE_LOG);
            info!("[{}:{}] {}", std::file!(), std::line!(), stringify!(#func));
            let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
            let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
            let proc_id = #proc_id;
            #( #def_statements )*
            let mut #result_name: #result_ty = Default::default();
            
            match proc_id.send(channel_sender) {
                Ok(()) => {}
                Err(e) => panic!("failed to send proc_id: {:?}", e),
            }
            #( #send_statements )*

            match channel_sender.flush_out() {
                Ok(()) => {}
                Err(e) => panic!("failed to flush_out: {:?}", e),
            }

            #( #recv_statements )*
            #( #assign_statements )*
            match #result_name.recv(channel_receiver) {
                Ok(()) => {}
                Err(e) => panic!("failed to receive #result_name: {:?}", e),
            }
            match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
            return #result_name;
        }
    };

    gen_fn.into()
}

#[proc_macro]
pub fn gen_hijack_async(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as HijackParser);

    let (proc_id, func, result, params) = (input.proc_id, input.func, input.result, input.params);

    let vars: Vec<Element> = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .enumerate()
        .map(|(i, param)| Element {
            name: format_ident!("var{}", i + 1),
            ty: param.ty.clone(),
            mode: ElementMode::Output,
        })
        .collect();

    // definition statements
    let def_statements = vars.iter().map(|var| {
        let name = &var.name;
        let ty = &var.ty;
        quote! { let mut #name: #ty = Default::default(); }
    });

    // send parameters
    let send_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            quote! {
                match #name.send(channel_sender) {
                    Ok(()) => {}
                    Err(e) => panic!("failed to send #name: {:?}", e),
                }
            }
        });

    let params = params.iter().map(|param| {
        let name = &param.name;
        let ty = &param.ty;
        match param.mode {
            ElementMode::Input => quote! { #name: #ty },
            ElementMode::Output => quote! { #name: *mut #ty },
        }
    });
    let result_ty = &result.ty;
    let result_ty_str = quote!(#result_ty).to_string();
    let tmp = result_ty_str.clone();
    let success_status = get_success_status(tmp.as_str());
    let result = result_ty_str + "::" + success_status;
    let result: Type = parse_str(&result).unwrap();
    let gen_fn = quote! {
        #[no_mangle]
        pub extern "C" fn #func(#(#params),*) -> #result_ty {
            assert_eq!(true, *ENABLE_LOG);
            info!("[{}:{}] {}", std::file!(), std::line!(), stringify!(#func));
            let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
            let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
            let proc_id = #proc_id;
            #( #def_statements )*
            
            match proc_id.send(channel_sender) {
                Ok(()) => {}
                Err(e) => panic!("failed to send proc_id: {:?}", e),
            }
            #( #send_statements )*
            match channel_sender.flush_out() {
                Ok(()) => {}
                Err(e) => panic!("failed to flush_out: {:?}", e),
            }
            return #result;
        }
    };

    gen_fn.into()
}

#[proc_macro]
pub fn gen_hijack_local(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as HijackParser);

    let (proc_id, func, result, params) = (input.proc_id, input.func, input.result, input.params);

    let vars: Vec<Element> = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .enumerate()
        .map(|(i, param)| Element {
            name: format_ident!("var{}", i + 1),
            ty: param.ty.clone(),
            mode: ElementMode::Output,
        })
        .collect();
    
    assert!(vars.len() == 1);

    // definition statements
    let def_statements = vars.iter().map(|var| {
        let name = &var.name;
        let ty = &var.ty;
        quote! { let mut #name: #ty = Default::default(); }
    });

    // send parameters
    let send_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            quote! {
                match #name.send(channel_sender) {
                    Ok(()) => {}
                    Err(e) => panic!("failed to send #name: {:?}", e),
                }
            }
        });

    // receive vars
    let recv_statements = vars.iter().map(|var| {
        let name = &var.name;
        quote! {
            match #name.recv(channel_receiver) {
                Ok(()) => {}
                Err(e) => panic!("failed to receive #name: {:?}", e),
            }
        }
    });

    let get_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output).map(|param| {
            let name = &param.name;
            quote! {
                if let Some(val) = get_local_info(proc_id as usize) {
                    unsafe { *#name = val as i32; }
                    return cudaError_t::cudaSuccess;
                }
            }
        });

    let add_statements = vars.iter().map(|var| {
        let name = &var.name;
        quote! {
            add_local_info(proc_id as usize, #name as usize);
        }
    });

    // assign vars to params
    let assign_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .zip(vars.iter())
        .map(|(param, var)| {
            let param_name = &param.name;
            let var_name = &var.name;
            quote! { unsafe { *#param_name = #var_name; } }
        });

    let params = params.iter().map(|param| {
        let name = &param.name;
        let ty = &param.ty;
        match param.mode {
            ElementMode::Input => quote! { #name: #ty },
            ElementMode::Output => quote! { #name: *mut #ty },
        }
    });
    let result_name = &result.name;
    let result_ty = &result.ty;
    let gen_fn = quote! {
        #[no_mangle]
        pub extern "C" fn #func(#(#params),*) -> #result_ty {
            assert_eq!(true, *ENABLE_LOG);
            info!("[{}:{}] {}", std::file!(), std::line!(), stringify!(#func));
            let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
            let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
            let proc_id = #proc_id;
            #( #def_statements )*
            let mut #result_name: #result_ty = Default::default();

            #( #get_statements )*
            
            match proc_id.send(channel_sender) {
                Ok(()) => {}
                Err(e) => panic!("failed to send proc_id: {:?}", e),
            }
            #( #send_statements )*
            match channel_sender.flush_out() {
                Ok(()) => {}
                Err(e) => panic!("failed to flush_out: {:?}", e),
            }

            #( #recv_statements )*
            #( #assign_statements )*
            match #result_name.recv(channel_receiver) {
                Ok(()) => {}
                Err(e) => panic!("failed to receive #result_name: {:?}", e),
            }
            match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
            #( #add_statements )*
            return #result_name;
        }
    };

    gen_fn.into()
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The collection of the procedural macro to generate Rust functions for server usage.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate execution functions for the server dispatcher.
///
/// To use this macro, annotate a call to `gen_exe` with the desired function name as the first
/// string literal argument, followed by the types of the return and parameters as string literals.
///
/// ### Example
/// We have a function `cudaSetDevice` with the following signature:
///
/// ```ignore
/// pub fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
/// ```
///
/// To use this macro generating a helper function for the server dispatcher, we can write:
///
/// ```ignore
/// gen_exe!("cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
/// ```
///
/// This invocation generates a function `cudaSetDeviceExe` with `channel_sender` and `channel_receiver` for communication.
/// `cudaSetDeviceExe` will be called by the server dispatcher to execute the native `cudaSetDevice` function and send the result back to the client.
///
/// Specifically, the function is expanded as:
///
/// ```ignore
/// pub fn cudaSetDeviceExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
///     info!("[{}:{}] cudaSetDevice", std::file!(), std::line!());
///     let mut param1: ::std::os::raw::c_int = Default::default();
///     param1.recv(channel_receiver).unwrap();
///     let result = unsafe { cudaSetDevice(param1) };
///     result.send(channel_sender).unwrap();
///     channel_sender.flush_out().unwrap();
/// }
/// ```
///
#[proc_macro]
pub fn gen_exe(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ExeParser);

    let (func, result, params) = (input.func, input.result, input.params);
    let func_exe = Ident::new(&format!("{}Exe", func), func.span());

    // definition statements
    let def_statements = params.iter().map(|param| {
        let name = &param.name;
        let ty = &param.ty;
        quote! { let mut #name: #ty = Default::default(); }
    });

    // receive parameters
    let recv_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            quote! { #name.recv(channel_receiver).unwrap(); }
        });

    // execution statement
    let result_name = &result.name;
    let exec_statement = {
        let api_index: u64 = get_api_index(&quote!(#func).to_string());
        let result_ty = &result.ty;
        let params = params.iter().map(|param| {
            let name = &param.name;
            quote! {
                get_address(&#name) as usize,
                #name.mem_size() as usize,
            }
        });
        let params = quote! { vec![ #(#params)* ] };
        quote! {
            let #result_name: #result_ty = {
                #result_ty::from(pos_process(
                    POS_CUDA_WS.lock().unwrap().get_ptr(),
                    #api_index,
                    0u64,
                    #params
                ))
            };
        }
    };

    // send result
    let send_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .map(|param| {
            let name = &param.name;
            quote! { #name.send(channel_sender).unwrap(); }
        });

    let gen_fn = quote! {
        pub fn #func_exe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
            info!("[{}:{}] {}", std::file!(), std::line!(), stringify!(#func));
            #( #def_statements )*
            #( #recv_statements )*
            match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e)
            }

            #exec_statement
            #( #send_statements )*
            #result_name.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    };

    gen_fn.into()
}

#[proc_macro]
pub fn gen_exe_async(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ExeParser);

    let (func, result, params) = (input.func, input.result, input.params);
    let func_exe = Ident::new(&format!("{}Exe", func), func.span());

    // definition statements
    let def_statements = params.iter().map(|param| {
        let name = &param.name;
        let ty = &param.ty;
        quote! { let mut #name: #ty = Default::default(); }
    });

    // receive parameters
    let recv_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            quote! { #name.recv(channel_receiver).unwrap(); }
        });

    // execution statement
    let result_name = &result.name;
    let exec_statement = {
        let api_index: u64 = get_api_index(&quote!(#func).to_string());
        let result_ty = &result.ty;
        let params = params.iter().map(|param| {
            let name = &param.name;
            quote! {
                get_address(&#name) as usize,
                #name.mem_size() as usize,
            }
        });
        let params = quote! { vec![ #(#params)* ] };
        quote! {
            let #result_name: #result_ty = {
                #result_ty::from(pos_process(
                        POS_CUDA_WS.lock().unwrap().get_ptr(),
                        #api_index,
                        0u64,
                        #params
                    ))
            };
        }
    };

    assert!(params.iter().filter(|param| param.mode == ElementMode::Output).count() == 0);
    let gen_fn = quote! {
        pub fn #func_exe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
            info!("[{}:{}] {}", std::file!(), std::line!(), stringify!(#func));
            #( #def_statements )*
            #( #recv_statements )*
            match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e)
            }
            #exec_statement
        }
    };
    gen_fn.into()
}

#[proc_macro]
pub fn gen_unimplement(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as UnimplementParser);

    let (func, result, params) = (input.func, input.result, input.params);

    let func_str = func.to_string();

    let result_ty = &result.ty;

    let params = params.iter().map(|param| {
        let name = &param.name;
        let ty = &param.ty;
        match param.mode {
            ElementMode::Input => quote! { #name: #ty },
            ElementMode::Output => quote! { #name: *mut #ty },
        }
    });

    let gen_fn = quote! {
        #[no_mangle]
        pub extern "C" fn #func(#(#params),*) -> #result_ty {
            unimplemented!("{}", #func_str);
        }
    };

    gen_fn.into()
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Resevation.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate a Rust function for serializing parameters to a buffer.
///
/// ### Example
/// To use this macro, annotate a call to `gen_serialize` with the desired function name as the first
/// string literal argument, followed by the types of the parameters as string literals.
///
/// gen_serialize!("my_function", "i32", "String");
///
/// This invocation generates a function `my_function` with two parameters: the first of type `i32` and
/// the second of type `String` (with a `buffer`` defined in `network`).
/// When `my_function` is called with values for these parameters, it serializes
/// each parameter's into the buffer.
///
/// Specifically, the function is defined as:
///
/// fn my_function(buf : &mut RawBuffer, a1: i32, a2: String) -> Result<(),BufferError> {
///    ...
/// }
///
#[proc_macro]
pub fn gen_serialize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::AttributeArgs);

    let fn_name = match &input[0] {
        syn::NestedMeta::Lit(syn::Lit::Str(lit_str)) => {
            Ident::new(&lit_str.value(), lit_str.span())
        }
        _ => panic!("Expected first argument to be a string literal for function name"),
    };

    let params: Vec<_> = input
        .iter()
        .skip(1)
        .enumerate()
        .map(|(i, arg)| {
            let ty = match arg {
                syn::NestedMeta::Lit(syn::Lit::Str(lit_str)) => {
                    lit_str.parse::<syn::Type>().expect("Expected valid type")
                }
                _ => panic!("Expected type as a string literal"),
            };
            let param_name = format_ident!("a{}", i + 1);
            quote! { #param_name: &#ty }
        })
        .collect();

    let param_names: Vec<_> = (1..=params.len())
        .map(|i| format_ident!("a{}", i))
        .collect();

    let serialize_statements = param_names.iter().map(|name| {
        //        quote! { println!("[{}] {}: {:?}", stringify!(#fn_name), stringify!(#name), #name); }
        quote! { unsafe { buf.append(#name)? }; }
    });

    let gen_fn = quote! {
        fn #fn_name(buf : &mut RawBuffer, #(#params),*) -> Result<(),BufferError> {
            #( #serialize_statements )*
            Ok(())
        }
    };

    gen_fn.into()
}

/// The code is similar to gen_serialize, but it generates a Rust function for deserializing parameters from a buffer.
#[proc_macro]
pub fn gen_deserialize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::AttributeArgs);

    let fn_name = match &input[0] {
        syn::NestedMeta::Lit(syn::Lit::Str(lit_str)) => {
            Ident::new(&lit_str.value(), lit_str.span())
        }
        _ => panic!("Expected first argument to be a string literal for function name"),
    };

    let params: Vec<_> = input
        .iter()
        .skip(1)
        .enumerate()
        .map(|(i, arg)| {
            let ty = match arg {
                syn::NestedMeta::Lit(syn::Lit::Str(lit_str)) => {
                    lit_str.parse::<syn::Type>().expect("Expected valid type")
                }
                _ => panic!("Expected type as a string literal"),
            };
            let param_name = format_ident!("a{}", i + 1);
            quote! { #param_name: &mut #ty }
        })
        .collect();

    let param_names: Vec<_> = (1..=params.len())
        .map(|i| format_ident!("a{}", i))
        .collect();

    let deserialize_statements = param_names.iter().map(|name| {
        quote! { unsafe { *(#name) = buf.extract()? }; }
    });

    let gen_fn = quote! {
        fn #fn_name(buf : &mut RawBuffer, #(#params),*) -> Result<(),BufferError> {
            #( #deserialize_statements )*
            Ok(())
        }
    };

    gen_fn.into()
}
