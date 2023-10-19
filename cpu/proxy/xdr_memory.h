#ifndef XDR_MEMORY_H
#define XDR_MEMORY_H

#include <rpc/types.h>
#include <rpc/xdr.h>
#include <sys/types.h>
#include <vector>

typedef int bool_t;
typedef unsigned int u_int;
typedef signed int int32_t;

class XDRMemory
{
public:
    XDRMemory();
    ~XDRMemory();

    int Size();
    void Resize(int size);
    char *Data();
    void Clear();

    // inner XDR ops
    bool_t Getlong(long *lp);
    bool_t Putlong(const long *lp);
    bool_t Getbytes(char *addr, u_int len);
    bool_t Putbytes(const char *addr, u_int len);
    u_int Getpos();
    bool_t Setpos(u_int pos);
    int32_t *Inline(u_int len);

private:
    std::vector<char> buffer;
    u_int position;
};

XDR *new_xdrmemory(enum xdr_op op);
void destroy_xdrmemory(XDR **xdrs);

#endif // XDR_MEMORY_H
