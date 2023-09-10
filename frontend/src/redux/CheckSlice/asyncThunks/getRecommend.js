import { createAsyncThunk } from "@reduxjs/toolkit";
import { CheckActions } from "../slice/checkSlice";

export const getRecommend = createAsyncThunk(
  'check/getRecommend',
  async(list, ThunkApi) => {
    try {
      const url = 'http://127.0.0.1:5000/predict_receipt';
      const body = JSON.stringify(list);
      const response = await fetch(url, {
        method: "POST",
        cache: "no-cache",
        mode: 'cors',
        headers: {
          "Content-Type": "application/json",
          "Accept" : "application/json"
        },
        body: body
      });
      const data = await response.json();
      ThunkApi.dispatch(CheckActions.setRecommend(data));
    } catch(e) {
      console.log(e);
      return ThunkApi.rejectWithValue('Произошла непредивденная ошибка')
    }
  }
)