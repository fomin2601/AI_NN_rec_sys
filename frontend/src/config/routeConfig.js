import CheckPage from "../pages/CheckPage/CheckPage";
import MainPage from "../pages/MainPage/MainPage";

export const routeConfig = [
  {
    element: <MainPage />,
    path: '/'
  },
  {
    element: <CheckPage />,
    path: '/check'
  }
];