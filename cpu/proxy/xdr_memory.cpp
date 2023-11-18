#include "xdr_memory.h"
#include <iostream>

XDRMemory::XDRMemory() : position(0)
{
    // Constructor with an initially empty buffer
}

XDRMemory::~XDRMemory()
{
    // Destructor
}

int XDRMemory::Size() { return buffer.size(); }

char *XDRMemory::Data() { return buffer.data(); }

void XDRMemory::Resize(int size)
{
    buffer.resize(size);
    position = 0;
}

void XDRMemory::Clear()
{
    buffer.clear();
    position = 0;
}

std::vector<char>& XDRMemory::GetBuffer() {
    return buffer;
}

bool_t XDRMemory::Getlong(long *lp)
{
    if (position + sizeof(long) <= buffer.size()) {
        *lp = *reinterpret_cast<long *>(&buffer[position]);
        position += sizeof(long);
        return 1;
    }
    return 0;
}

bool_t XDRMemory::Putlong(const long *lp)
{
    // std::cout << "Putlong: " << *lp << std::endl;
    // Calculate the new buffer size if necessary
    if (position + sizeof(long) > buffer.size()) {
        buffer.resize(position + sizeof(long));
        // std::cout << "Resize: " << buffer.size() << std::endl;
    }
    *reinterpret_cast<long *>(&buffer[position]) = *lp;
    position += sizeof(long);
    return 1;
}

bool_t XDRMemory::Getbytes(char *addr, u_int len)
{
    if (position + len <= buffer.size()) {
        std::copy(buffer.begin() + position, buffer.begin() + position + len,
                  addr);
        position += len;
        return true;
    }
    return false;
}

bool_t XDRMemory::Putbytes(const char *addr, u_int len)
{
    // Calculate the new buffer size if necessary
    if (position + len > buffer.size()) {
        buffer.resize(position + len);
    }
    std::copy(addr, addr + len, buffer.begin() + position);
    position += len;
    return true;
}

u_int XDRMemory::Getpos() { return position; }

bool_t XDRMemory::Setpos(u_int pos)
{
    if (pos <= buffer.size()) {
        position = pos;
        return 1;
    }
    return 0;
}

int32_t *XDRMemory::Inline(u_int len)
{
    // Calculate the new buffer size if necessary
    if (position + len * sizeof(int32_t) > buffer.size()) {
        buffer.resize(position + len * sizeof(int32_t));
    }
    int32_t *ptr = reinterpret_cast<int32_t *>(&buffer[position]);
    position += len * sizeof(int32_t);
    return ptr;
}

// XDR-compatible

extern "C" {
static void xdrmemory_destroy(XDR *);
static bool_t xdrmemory_getlong(XDR *, long *);
static bool_t xdrmemory_putlong(XDR *, const long *);
static bool_t xdrmemory_getbytes(XDR *, char *, u_int);
static bool_t xdrmemory_putbytes(XDR *, const char *, u_int);
static u_int xdrmemory_getpos(XDR *);
static bool_t xdrmemory_setpos(XDR *, u_int);
static int32_t *xdrmemory_inline(XDR *, u_int);
}

static const struct xdr_ops xdrmemory_ops = {
    xdrmemory_getlong,  xdrmemory_putlong, xdrmemory_getbytes,
    xdrmemory_putbytes, xdrmemory_getpos,  xdrmemory_setpos,
    xdrmemory_inline,   xdrmemory_destroy
};

XDR *new_xdrmemory(enum xdr_op op)
{
    XDR *xdrs = reinterpret_cast<XDR *>(malloc(sizeof(XDR)));
    xdrs->x_op = op;
    xdrs->x_ops = &xdrmemory_ops;
    XDRMemory *xdrmemory = new XDRMemory();
    xdrs->x_private = reinterpret_cast<char *>(xdrmemory);
    xdrs->x_public = xdrs->x_base = NULL;
    xdrs->x_handy = 0;
}

void destroy_xdrmemory(XDR **xdrs)
{
    if (*xdrs) {
        xdrmemory_destroy(*xdrs);
        free(*xdrs);
        *xdrs = NULL;
    }
}

static void xdrmemory_destroy(XDR *xdrs)
{
    if (xdrs->x_private) {
        delete reinterpret_cast<XDRMemory *>(xdrs->x_private);
        xdrs->x_private = NULL;
    }
}

static bool_t xdrmemory_getlong(XDR *xdrs, long *lp)
{
    return reinterpret_cast<XDRMemory *>(xdrs->x_private)->Getlong(lp);
}

static bool_t xdrmemory_putlong(XDR *xdrs, const long *lp)
{
    return reinterpret_cast<XDRMemory *>(xdrs->x_private)->Putlong(lp);
}

static bool_t xdrmemory_getbytes(XDR *xdrs, char *addr, u_int len)
{
    return reinterpret_cast<XDRMemory *>(xdrs->x_private)->Getbytes(addr, len);
}

static bool_t xdrmemory_putbytes(XDR *xdrs, const char *addr, u_int len)
{
    return reinterpret_cast<XDRMemory *>(xdrs->x_private)->Putbytes(addr, len);
}

static u_int xdrmemory_getpos(XDR *xdrs)
{
    return reinterpret_cast<XDRMemory *>(xdrs->x_private)->Getpos();
}

static bool_t xdrmemory_setpos(XDR *xdrs, u_int pos)
{
    return reinterpret_cast<XDRMemory *>(xdrs->x_private)->Setpos(pos);
}

static int32_t *xdrmemory_inline(XDR *xdrs, u_int len)
{
    return reinterpret_cast<XDRMemory *>(xdrs->x_private)->Inline(len);
}
