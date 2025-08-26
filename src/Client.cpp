#include <arpa/inet.h>
#include <string>
#include <unistd.h>

#include "Client.hpp"

Client::Client() {
    sock = socket(AF_INET, SOCK_STREAM, 0);

    if(sock != -1) {
        struct sockaddr_in server;
        server.sin_family = AF_INET;
        server.sin_port = htons(8080);
        server.sin_addr.s_addr = inet_addr("127.0.0.1");

        connect(sock, (struct sockaddr *)&server, sizeof(server));
    }
}

Client::~Client() {
    close(sock);
}

void Client::write(std::string message) {
    send(sock, message.c_str(), message.size(), 0);
}
