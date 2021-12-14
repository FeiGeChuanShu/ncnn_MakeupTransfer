// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.makeup;
import android.os.Build;
import android.Manifest;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.media.ExifInterface;
import android.graphics.Matrix;
import android.content.pm.PackageManager;
import java.io.File;
import java.io.IOException;
import java.io.FileNotFoundException;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.provider.MediaStore;

import android.os.Environment;
public class MainActivity extends Activity
{
    private static final int SELECT_REFERENCE_IMAGE = 1;
    private static final int SELECT_TARGET_IMAGE = 2;
    private int style_type = 0;
    private ImageView imageView1;
    private ImageView imageView2;
    private ImageView imageView3;
    private Bitmap targetImage = null;
    private Bitmap referenceImage = null;

    private Makeup makeup = new Makeup();
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        //requestPermission();
        boolean ret_init = makeup.Init(getAssets());
        if (!ret_init)
        {
            Log.e("MainActivity", "makeup Init failed");
        }

        imageView1 = (ImageView) findViewById(R.id.imageView1);
        imageView2 = (ImageView) findViewById(R.id.imageView2);
        imageView3 = (ImageView) findViewById(R.id.imageView3);
        Button buttonTargetImage = (Button) findViewById(R.id.buttonTargetImage);
        buttonTargetImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_TARGET_IMAGE);
            }
        });
        Button buttonReferenceImage = (Button) findViewById(R.id.buttonReferenceImage);
        buttonReferenceImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_REFERENCE_IMAGE);
            }
        });
        Button buttonDetect = (Button) findViewById(R.id.buttonDetect);
        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (targetImage == null || referenceImage == null)
                    return;

                getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                new Thread(new Runnable() {
                    public void run() {
                        final Bitmap styledImage = runStyleTransfer(false);
                        imageView2.post(new Runnable() {
                            public void run() {
                                imageView2.setImageBitmap(styledImage);
                                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                            }
                        });
                    }
                }).start();
            }
        });

        Button buttonDetectGPU = (Button) findViewById(R.id.buttonDetectGPU);
        buttonDetectGPU.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (targetImage == null || referenceImage == null)
                    return;

                getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                new Thread(new Runnable() {
                    public void run() {
                        final Bitmap styledImage = runStyleTransfer(true);
                        imageView2.post(new Runnable() {
                            public void run() {
                                imageView2.setImageBitmap(styledImage);
                                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                            }
                        });
                    }
                }).start();
            }
        });

    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case SELECT_REFERENCE_IMAGE:
                if (resultCode == RESULT_OK && null != data) {
                    Uri selectedImage = data.getData();
                    try {
                        if (requestCode == SELECT_REFERENCE_IMAGE) {
                            referenceImage = decodeUri(selectedImage);

                            imageView3.setImageBitmap(referenceImage);
                        }
                    }
                    catch (FileNotFoundException e) {
                        Log.e("MainActivity", "FileNotFoundException");
                        return;
                    }
                }
                break;
            case SELECT_TARGET_IMAGE:
                if (resultCode == RESULT_OK && null != data) {
                    Uri selectedImage = data.getData();
                    try {
                        if (requestCode == SELECT_TARGET_IMAGE) {
                            targetImage = decodeUri(selectedImage);

                            imageView1.setImageBitmap(targetImage);
                        }
                    }
                    catch (FileNotFoundException e) {
                        Log.e("MainActivity", "FileNotFoundException");
                        return;
                    }
                }
            default:
                break;
        }
    }

    private Bitmap runStyleTransfer(boolean use_gpu)
    {
        Bitmap targetBitmapImage = targetImage.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap referenceBitmapImage = referenceImage.copy(Bitmap.Config.ARGB_8888, true);
        makeup.Process(targetBitmapImage,referenceBitmapImage, style_type, use_gpu);
        return targetBitmapImage;
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 400;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
               || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);

        // Rotate according to EXIF
        int rotate = 0;
        try
        {
            ExifInterface exif = new ExifInterface(getContentResolver().openInputStream(selectedImage));
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotate = 270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotate = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotate = 90;
                    break;
            }
        }
        catch (IOException e)
        {
            Log.e("MainActivity", "ExifInterface IOException");
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(rotate);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

}
