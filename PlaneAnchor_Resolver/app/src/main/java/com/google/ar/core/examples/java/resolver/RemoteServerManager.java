package com.google.ar.core.examples.java.resolver;

import android.os.Handler;
import android.util.Log;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;

public class RemoteServerManager {
    private String html = "";
    private Handler mHandler;

    private String serverIP = "";
    private int serverPort = 0;

    private Socket socket;

    private DataOutputStream dos;
    private DataInputStream dis;

    RemoteServerManager(String serverIP, int serverPort){
        Log.w("yunho-server", "RemoteServerManager");
        this.serverIP = serverIP;
        this.serverPort = serverPort;
    }

    void sendRoomCode(Long roomCode){
        mHandler = new Handler();
        Log.w("yunho-server", "start connect");
        Thread checkUpdate = new Thread() {
            public void run(){
                try{
                    socket = new Socket(serverIP, serverPort);
                    Log.w("yunho-server", "connect success");
                } catch (IOException e){
                    Log.w("yunho-server", "connect failed");
                    e.printStackTrace();
                }

                try{
                    dos = new DataOutputStream(socket.getOutputStream());
                    dis = new DataInputStream(socket.getInputStream());
                    dos.writeUTF("RoomCode/"+roomCode);
                } catch (IOException e){
                    e.printStackTrace();
                    Log.w("yunho-server", "buffer failed");
                }
                Log.w("yunho-server", "buffer successed");
            }
        };
        checkUpdate.start();
    }

    void setServerIP(String ip){
        this.serverIP = ip;
    }

    void setServerPort(int port){
        this.serverPort = port;
    }
}
