import { createSlice} from "@reduxjs/toolkit";
import { getRecommend } from "../asyncThunks/getRecommend";

const initialState = {
  inputId: '',
  items: {},
  itemsList: [],
  recLoading: false,
  recError: '',
  recItem: {}
}

export const checkSlice = createSlice({
  name: 'check',
  initialState,
  reducers: {
    setData: (state, action) => {
      state.items = action.payload;
    },
    changeInputId: (state, action) => {
      const {
        payload: inputId
      } = action;
      state.inputId = inputId;
      // console.log('items', items);
      state.inputItem = state.items[inputId] ? {item_id: parseInt(inputId), ...state.items[inputId]} : {};
    },
    addItemToList: (state, action) => {
      const { 
        id: itemId,
        inputItem
      } = action.payload;

      const itemIndex = state.itemsList.findIndex((item => item.item_id === parseInt(itemId)));

      if (itemIndex !== -1) {
        state.itemsList[itemIndex].quantity += 1;
      } else {
        state.itemsList.push({...inputItem, quantity: 1, item_id: parseInt(itemId)});
      }

      state.inputId = '';
      state.inputItem = {};
    },
    increaseQuantity: (state, action) => {
      const itemId = parseInt(action.payload);
      const itemIndex = state.itemsList.findIndex((item => item.item_id === itemId));

      state.itemsList[itemIndex].quantity += 1;
    },
    decreaseQuantity: (state, action) => {
      const itemId = parseInt(action.payload);
      const itemIndex = state.itemsList.findIndex((item => item.item_id === itemId));

      if (state.itemsList[itemIndex].quantity - 1 > 0) {
        state.itemsList[itemIndex].quantity = state.itemsList[itemIndex].quantity - 1;
      } else {
        state.itemsList.splice(itemIndex, 1);
      }
    },
    addRecommendToList: (state, action) => {
      const recItemId = parseInt(state.recItem.id);

      const itemIndex = state.itemsList.findIndex((item => item.item_id === recItemId));

      if (itemIndex !== -1) {
        state.itemsList[itemIndex].quantity += 1;
      } else {
        state.itemsList.push({...state.recItem, quantity: 1});
      }

    },
    clearData: (state, action) => {
      state.inputId = initialState.inputId;
      state.inputItem = {};
      state.itemsList = initialState.itemsList;
      state.recItem = initialState.recItem;
    },
    setRecommend: (state, action) => {
      const recId = action.payload;

      if (state.items[recId]) {
        state.recItem = {
          item_id: parseInt(recId),
          name: state.items[recId].name,
          price: state.items[recId].price
        };
      } else {
        state.recItem = {};
      }
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(getRecommend.pending, (state) => {
        state.recLoading = true;
        state.recError = '';
      })
      .addCase(getRecommend.fulfilled, (state) => {
        state.recLoading = false;
        state.recError = '';
      })
      .addCase(getRecommend.rejected, (state) => {
        state.recLoading = false;
        state.recError = 'Рекомендованный товар не найден';
      })
  }
})

export const { actions: CheckActions } = checkSlice;
export const { reducer: CheckReducer } = checkSlice;