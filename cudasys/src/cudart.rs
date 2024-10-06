use std::boxed::Box;
use super::*;
pub use crate::types::cudart::*;
include!("bindings/funcs/cudart.rs");

impl cudaError_t {
    pub fn from(value: i32) -> Self {
        match value {
            0 => cudaError_t::cudaSuccess,
            1 => cudaError_t::cudaErrorInvalidValue,
            2 => cudaError_t::cudaErrorMemoryAllocation,
            3 => cudaError_t::cudaErrorInitializationError,
            4 => cudaError_t::cudaErrorCudartUnloading,
            5 => cudaError_t::cudaErrorProfilerDisabled,
            6 => cudaError_t::cudaErrorProfilerNotInitialized,
            7 => cudaError_t::cudaErrorProfilerAlreadyStarted,
            8 => cudaError_t::cudaErrorProfilerAlreadyStopped,
            9 => cudaError_t::cudaErrorInvalidConfiguration,
            12 => cudaError_t::cudaErrorInvalidPitchValue,
            13 => cudaError_t::cudaErrorInvalidSymbol,
            16 => cudaError_t::cudaErrorInvalidHostPointer,
            17 => cudaError_t::cudaErrorInvalidDevicePointer,
            18 => cudaError_t::cudaErrorInvalidTexture,
            19 => cudaError_t::cudaErrorInvalidTextureBinding,
            20 => cudaError_t::cudaErrorInvalidChannelDescriptor,
            21 => cudaError_t::cudaErrorInvalidMemcpyDirection,
            22 => cudaError_t::cudaErrorAddressOfConstant,
            23 => cudaError_t::cudaErrorTextureFetchFailed,
            24 => cudaError_t::cudaErrorTextureNotBound,
            25 => cudaError_t::cudaErrorSynchronizationError,
            26 => cudaError_t::cudaErrorInvalidFilterSetting,
            27 => cudaError_t::cudaErrorInvalidNormSetting,
            28 => cudaError_t::cudaErrorMixedDeviceExecution,
            31 => cudaError_t::cudaErrorNotYetImplemented,
            32 => cudaError_t::cudaErrorMemoryValueTooLarge,
            34 => cudaError_t::cudaErrorStubLibrary,
            35 => cudaError_t::cudaErrorInsufficientDriver,
            36 => cudaError_t::cudaErrorCallRequiresNewerDriver,
            37 => cudaError_t::cudaErrorInvalidSurface,
            43 => cudaError_t::cudaErrorDuplicateVariableName,
            44 => cudaError_t::cudaErrorDuplicateTextureName,
            45 => cudaError_t::cudaErrorDuplicateSurfaceName,
            46 => cudaError_t::cudaErrorDevicesUnavailable,
            49 => cudaError_t::cudaErrorIncompatibleDriverContext,
            52 => cudaError_t::cudaErrorMissingConfiguration,
            53 => cudaError_t::cudaErrorPriorLaunchFailure,
            65 => cudaError_t::cudaErrorLaunchMaxDepthExceeded,
            66 => cudaError_t::cudaErrorLaunchFileScopedTex,
            67 => cudaError_t::cudaErrorLaunchFileScopedSurf,
            68 => cudaError_t::cudaErrorSyncDepthExceeded,
            69 => cudaError_t::cudaErrorLaunchPendingCountExceeded,
            98 => cudaError_t::cudaErrorInvalidDeviceFunction,
            100 => cudaError_t::cudaErrorNoDevice,
            101 => cudaError_t::cudaErrorInvalidDevice,
            102 => cudaError_t::cudaErrorDeviceNotLicensed,
            103 => cudaError_t::cudaErrorSoftwareValidityNotEstablished,
            127 => cudaError_t::cudaErrorStartupFailure,
            200 => cudaError_t::cudaErrorInvalidKernelImage,
            201 => cudaError_t::cudaErrorDeviceUninitialized,
            205 => cudaError_t::cudaErrorMapBufferObjectFailed,
            206 => cudaError_t::cudaErrorUnmapBufferObjectFailed,
            207 => cudaError_t::cudaErrorArrayIsMapped,
            208 => cudaError_t::cudaErrorAlreadyMapped,
            209 => cudaError_t::cudaErrorNoKernelImageForDevice,
            210 => cudaError_t::cudaErrorAlreadyAcquired,
            211 => cudaError_t::cudaErrorNotMapped,
            212 => cudaError_t::cudaErrorNotMappedAsArray,
            213 => cudaError_t::cudaErrorNotMappedAsPointer,
            214 => cudaError_t::cudaErrorECCUncorrectable,
            215 => cudaError_t::cudaErrorUnsupportedLimit,
            216 => cudaError_t::cudaErrorDeviceAlreadyInUse,
            217 => cudaError_t::cudaErrorPeerAccessUnsupported,
            218 => cudaError_t::cudaErrorInvalidPtx,
            219 => cudaError_t::cudaErrorInvalidGraphicsContext,
            220 => cudaError_t::cudaErrorNvlinkUncorrectable,
            221 => cudaError_t::cudaErrorJitCompilerNotFound,
            222 => cudaError_t::cudaErrorUnsupportedPtxVersion,
            223 => cudaError_t::cudaErrorJitCompilationDisabled,
            300 => cudaError_t::cudaErrorInvalidSource,
            301 => cudaError_t::cudaErrorFileNotFound,
            302 => cudaError_t::cudaErrorSharedObjectSymbolNotFound,
            303 => cudaError_t::cudaErrorSharedObjectInitFailed,
            304 => cudaError_t::cudaErrorOperatingSystem,
            400 => cudaError_t::cudaErrorInvalidResourceHandle,
            401 => cudaError_t::cudaErrorIllegalState,
            500 => cudaError_t::cudaErrorSymbolNotFound,
            600 => cudaError_t::cudaErrorNotReady,
            700 => cudaError_t::cudaErrorIllegalAddress,
            701 => cudaError_t::cudaErrorLaunchOutOfResources,
            702 => cudaError_t::cudaErrorLaunchTimeout,
            703 => cudaError_t::cudaErrorLaunchIncompatibleTexturing,
            704 => cudaError_t::cudaErrorPeerAccessAlreadyEnabled,
            705 => cudaError_t::cudaErrorPeerAccessNotEnabled,
            708 => cudaError_t::cudaErrorSetOnActiveProcess,
            709 => cudaError_t::cudaErrorContextIsDestroyed,
            710 => cudaError_t::cudaErrorAssert,
            711 => cudaError_t::cudaErrorTooManyPeers,
            712 => cudaError_t::cudaErrorHostMemoryAlreadyRegistered,
            713 => cudaError_t::cudaErrorHostMemoryNotRegistered,
            714 => cudaError_t::cudaErrorHardwareStackError,
            715 => cudaError_t::cudaErrorIllegalInstruction,
            716 => cudaError_t::cudaErrorMisalignedAddress,
            717 => cudaError_t::cudaErrorInvalidAddressSpace,
            718 => cudaError_t::cudaErrorInvalidPc,
            719 => cudaError_t::cudaErrorLaunchFailure,
            720 => cudaError_t::cudaErrorCooperativeLaunchTooLarge,
            800 => cudaError_t::cudaErrorNotPermitted,
            801 => cudaError_t::cudaErrorNotSupported,
            802 => cudaError_t::cudaErrorSystemNotReady,
            803 => cudaError_t::cudaErrorSystemDriverMismatch,
            804 => cudaError_t::cudaErrorCompatNotSupportedOnDevice,
            900 => cudaError_t::cudaErrorStreamCaptureUnsupported,
            901 => cudaError_t::cudaErrorStreamCaptureInvalidated,
            902 => cudaError_t::cudaErrorStreamCaptureMerge,
            903 => cudaError_t::cudaErrorStreamCaptureUnmatched,
            904 => cudaError_t::cudaErrorStreamCaptureUnjoined,
            905 => cudaError_t::cudaErrorStreamCaptureIsolation,
            906 => cudaError_t::cudaErrorStreamCaptureImplicit,
            907 => cudaError_t::cudaErrorCapturedEvent,
            908 => cudaError_t::cudaErrorStreamCaptureWrongThread,
            909 => cudaError_t::cudaErrorTimeout,
            910 => cudaError_t::cudaErrorGraphExecUpdateFailure,
            999 => cudaError_t::cudaErrorUnknown,
            10000 => cudaError_t::cudaErrorApiFailureBase,
            _ => panic!(format!("unsupported cudaError_t {}", value))
        }
    }
}

/// cudaStream_t is a pointer type, we just need to use usize to represent it.
/// It is not necessary to define a struct for it, as the struct is also just a placeholder.

#[cfg(test)]
mod tests{
    use super::*;
    use crate::FromPrimitive;
    use network::{
        ringbufferchannel::{META_AREA, LocalChannel},
        Channel,
    };

    #[test]
    fn test_num_derive() {
        let x: u32 = cudaError_t::cudaSuccess as u32;
        assert_eq!(x, 0);
        match cudaError_t::from_u32(1) {
            Some(v) => assert_eq!(v, cudaError_t::cudaErrorInvalidValue),
            None => panic!("failed to convert from u32"),
        }
    }

    #[test]
    fn test_cudaError_t_io() {
        let mut buffer: Channel =
            Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = cudaError_t::cudaErrorInvalidValue;
        let mut b = cudaError_t::cudaSuccess;
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_cudaStream_t_io() {
        let mut buffer: Channel =
            Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = 100usize as cudaStream_t;
        let mut b: cudaStream_t = 0usize as cudaStream_t;
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    pub fn cuda_ffi() {
        let mut device = 0;
        let mut device_num = 0;

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDeviceCount(&mut device_num as *mut i32) }
        {
            println!("device count: {}", device_num);
        } else {
            panic!("failed to get device count");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDevice(&mut device as *mut i32) } {
            assert_eq!(device, 0);
            println!("device: {}", device);
        } else {
            panic!("failed to get device");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaSetDevice(device_num - 1) } {
        } else {
            panic!("failed to set device");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDevice(&mut device as *mut i32) } {
            assert_eq!(device, device_num - 1);
            println!("device: {}", device);
        } else {
            panic!("failed to get device");
        }
    }
}
