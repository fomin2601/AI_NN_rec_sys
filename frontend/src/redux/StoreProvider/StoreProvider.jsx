import { configureStore } from "@reduxjs/toolkit";
import { CheckReducer } from "../CheckSlice/slice/checkSlice";
import { Provider } from "react-redux";

function createReduxStore(initialState) {
  const rootReducers = {
    check: CheckReducer
  }

  const store = configureStore({
    reducer: rootReducers,
    preloadedState: initialState
  })

  return store;
}

export const StoreProvider = (props) => {
  const {
    children,
    initialState
  } = props;

  const store = createReduxStore(initialState);

  return (
    <Provider store={store}>
      {children}
    </Provider>
  )
}