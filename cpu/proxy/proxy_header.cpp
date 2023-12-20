#include "proxy_header.h"

ProxyHeader::ProxyHeader(int proc_id, int device_id, int retrieve_flag)
{
    this->proc_id = proc_id;
    this->device_id = device_id;
    this->retrieve_flag = retrieve_flag;
}

int ProxyHeader::get_proc_id() { return this->proc_id; }

int ProxyHeader::get_device_id() { return this->device_id; }

int ProxyHeader::get_retrieve_flag() { return this->retrieve_flag; }

void ProxyHeader::set_proc_id(int proc_id) { this->proc_id = proc_id; }

void ProxyHeader::set_device_id(int device_id) { this->device_id = device_id; }

void ProxyHeader::set_retrieve_flag(int retrieve_flag)
{
    this->retrieve_flag = retrieve_flag;
}
