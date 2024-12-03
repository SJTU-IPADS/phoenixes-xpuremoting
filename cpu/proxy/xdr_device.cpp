#include "xdr_device.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <rpc/xdr.h>

XDRDevice::XDRDevice(DeviceBuffer *buffer)
    : buffer_(buffer), position(0), mask(0)
{
}

XDRDevice::~XDRDevice() { buffer_ = nullptr; }

void XDRDevice::SetBuffer(DeviceBuffer *buffer) { buffer_ = buffer; }

DeviceBuffer *XDRDevice::GetBuffer() { return buffer_; }

void XDRDevice::SetMask(int m) { mask = m; }

int XDRDevice::GetMask() { return mask; }

bool_t XDRDevice::Getlong(long *lp)
{
    int ret = 0;
    assert(buffer_ != nullptr);
    if (mask == 1) {
        *lp = 0;
        return true;
    }

    while(ret == 0)
        ret = buffer_->getBytes((char *)lp, sizeof(long));

    if (ret < 0) {
        return false;
    }
    position += sizeof(long);
    return true;
}

bool_t XDRDevice::Putlong(const long *lp)
{
    assert(buffer_ != nullptr);
    if (mask == 1) {
        return true;
    }
    if (buffer_->putBytes((char *)lp, sizeof(long)) < 0) {
        return false;
    }
    position += sizeof(long);
    return true;
}

bool_t XDRDevice::Getbytes(char *addr, u_int len)
{
    int ret = 0;
    assert(buffer_ != nullptr);
    if (mask == 1) {
        memset(addr, 0, len);
        return true;
    }

    while(ret == 0)
        ret = buffer_->getBytes(addr, len);

    if(ret < 0)
        return false;
    
    position += len;
    return true;
}

bool_t XDRDevice::Putbytes(const char *addr, u_int len)
{
    assert(buffer_ != nullptr);
    if (mask == 1) {
        return true;
    }
    if (buffer_->putBytes(addr, len) < 0) {
        return false;
    }
    position += len;
    return true;
}

u_int XDRDevice::Getpos()
{
    if (mask == 1) {
        return 0;
    }
    return position;
}

bool_t XDRDevice::Setpos(u_int pos)
{
    if (mask == 1) {
        return true;
    }
    position = pos;
    return true;
}

int32_t *XDRDevice::Inline(u_int len)
{
    std::cout << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
              << std::endl;
    exit(1);
}

// XDR-compatible

extern "C" {
static void xdrdevice_destroy(XDR *);
static bool_t xdrdevice_getlong(XDR *, long *);
static bool_t xdrdevice_putlong(XDR *, const long *);
static bool_t xdrdevice_getbytes(XDR *, char *, u_int);
static bool_t xdrdevice_putbytes(XDR *, const char *, u_int);
static u_int xdrdevice_getpos(XDR *);
static bool_t xdrdevice_setpos(XDR *, u_int);
static int32_t *xdrdevice_inline(XDR *, u_int);
}

static const struct xdr_ops xdrdevice_ops = {
    xdrdevice_getlong,  xdrdevice_putlong, xdrdevice_getbytes,
    xdrdevice_putbytes, xdrdevice_getpos,  xdrdevice_setpos,
    xdrdevice_inline,   xdrdevice_destroy
};

XDR *new_xdrdevice(enum xdr_op op)
{
    XDR *xdrs = reinterpret_cast<XDR *>(malloc(sizeof(XDR)));
    if (xdrs == nullptr) {
        return NULL;
    }
    xdrs->x_op = op;
    xdrs->x_ops = &xdrdevice_ops;
    XDRDevice *xdrdevice = new XDRDevice();
    xdrs->x_private = reinterpret_cast<char *>(xdrdevice);
    xdrs->x_public = xdrs->x_base = NULL;
    xdrs->x_handy = 0;
    return xdrs;
}

void destroy_xdrdevice(XDR **xdrs)
{
    if (xdrs == NULL || *xdrs == NULL) {
        return;
    }
    xdrdevice_destroy(*xdrs);
    free(*xdrs);
    *xdrs = NULL;
}

static void xdrdevice_destroy(XDR *xdrs)
{
    if (xdrs->x_private) {
        delete reinterpret_cast<XDRDevice *>(xdrs->x_private);
        xdrs->x_private = nullptr;
    }
}

static bool_t xdrdevice_getlong(XDR *xdrs, long *lp)
{
    return reinterpret_cast<XDRDevice *>(xdrs->x_private)->Getlong(lp);
}

static bool_t xdrdevice_putlong(XDR *xdrs, const long *lp)
{
    return reinterpret_cast<XDRDevice *>(xdrs->x_private)->Putlong(lp);
}

static bool_t xdrdevice_getbytes(XDR *xdrs, char *addr, u_int len)
{
    return reinterpret_cast<XDRDevice *>(xdrs->x_private)->Getbytes(addr, len);
}

static bool_t xdrdevice_putbytes(XDR *xdrs, const char *addr, u_int len)
{
    return reinterpret_cast<XDRDevice *>(xdrs->x_private)->Putbytes(addr, len);
}

static u_int xdrdevice_getpos(XDR *xdrs)
{
    return reinterpret_cast<XDRDevice *>(xdrs->x_private)->Getpos();
}

static bool_t xdrdevice_setpos(XDR *xdrs, u_int pos)
{
    return reinterpret_cast<XDRDevice *>(xdrs->x_private)->Setpos(pos);
}

static int32_t *xdrdevice_inline(XDR *xdrs, u_int len)
{
    return reinterpret_cast<XDRDevice *>(xdrs->x_private)->Inline(len);
}
