#ifndef PROXY_HEADER_H
#define PROXY_HEADER_H

// the proxy header plain struct
class ProxyHeader
{
public:
    ProxyHeader(int proc_id = 0, int device_id = 0);
    int get_proc_id();
    int get_device_id();
    void set_proc_id(int proc_id);
    void set_device_id(int device_id);

private:
    int proc_id;
    int device_id;
};

#endif // PROXY_HEADER_H
