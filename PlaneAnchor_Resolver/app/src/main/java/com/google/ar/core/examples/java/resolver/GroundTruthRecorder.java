package com.google.ar.core.examples.java.resolver;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import com.intel.realsense.librealsense.Config;
import com.intel.realsense.librealsense.DeviceList;
import com.intel.realsense.librealsense.DeviceListener;
import com.intel.realsense.librealsense.FrameSet;
import com.intel.realsense.librealsense.GLRsSurfaceView;
import com.intel.realsense.librealsense.Pipeline;
import com.intel.realsense.librealsense.PipelineProfile;
import com.intel.realsense.librealsense.RsContext;
import com.intel.realsense.librealsense.StreamType;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

public class GroundTruthRecorder {
    private static final String TAG = "librs recording example";

    private boolean mIsStreaming = false;
    private final Handler mHandler = new Handler();

    private Pipeline mPipeline;
    private Context mContext;
    private RsContext mRsContext;

    public GroundTruthRecorder(Context context){
        Context mContext;
        mContext = context;
        RsContext.init(((HelloArActivity) mContext).getApplicationContext());
        mRsContext = new RsContext();
        mRsContext.setDevicesChangedCallback(mListener);
        mPipeline = new Pipeline();
    }


    private String getFilePath(String fileName){
        File folder = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getPath() + File.separator + "rs_bags");
        folder.mkdir();
        File file = new File(folder, fileName + ".bag");
        return file.getAbsolutePath();
    }


    private DeviceListener mListener = new DeviceListener() {
        @Override
        public void onDeviceAttach() {
            ((HelloArActivity) mContext).messageSnackbarHelper.showMessageWithDismiss(
                    (HelloArActivity) mContext, "Device Attached");
        }

        @Override
        public void onDeviceDetach() {
            recordBagStop();
        }
    };


    Runnable mStreaming = new Runnable() {
        @Override
        public void run() {
            try {
                FrameSet frames = mPipeline.waitForFrames();
            }
            catch (Exception e) {
                Log.e(TAG, "streaming, error: " + e.getMessage());
            }
        }
    };


    public synchronized void recordBagStart() {
        if(mIsStreaming)
            return;
        try{
            Log.d(TAG, "try start streaming");
            try(Config cfg = new Config()) {
                cfg.enableStream(StreamType.DEPTH, 640, 480);
                cfg.enableStream(StreamType.COLOR, 640, 480);
                SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss");
                String currentDateAndTime = sdf.format(new Date());
                cfg.enableRecordToFile(getFilePath(currentDateAndTime));
                try(PipelineProfile pp = mPipeline.start(cfg)){}
            }
            mIsStreaming = true;
            mHandler.post(mStreaming);
            Log.d(TAG, "streaming started successfully");
        } catch (Exception e) {
            Log.d(TAG, "failed to start streaming");
        }
    }


    public synchronized void recordBagStart(String fileName) {
        if(mIsStreaming)
            return;
        try{
            Log.d(TAG, "try start streaming");
            try(Config cfg = new Config()) {
                cfg.enableStream(StreamType.DEPTH, 640, 480);
                cfg.enableStream(StreamType.COLOR, 640, 480);
                cfg.enableRecordToFile(getFilePath(fileName));
                try(PipelineProfile pp = mPipeline.start(cfg)){}
            }
            mIsStreaming = true;
            mHandler.post(mStreaming);
            Log.d(TAG, "streaming started successfully");
        } catch (Exception e) {
            Log.d(TAG, "failed to start streaming");
        }
    }


    public synchronized void recordBagStop() {
        if(!mIsStreaming)
            return;
        try {
            Log.d(TAG, "try stop streaming");
            mIsStreaming = false;
            mHandler.removeCallbacks(mStreaming);
            mPipeline.stop();
            Log.d(TAG, "streaming stopped successfully");
        }  catch (Exception e) {
            Log.d(TAG, "failed to stop streaming");
            mPipeline = null;
        }
    }
}
