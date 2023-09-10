import React, { useEffect, useMemo, useState } from 'react';
import cls from './CheckPage.module.scss';
import CheckHeader from '../../components/CheckHeader/CheckHeader';
import CheckList from '../../components/CheckList/CheckList';
import { useDispatch, useSelector } from 'react-redux';
import { getCheckItemsList } from '../../redux/CheckSlice/selectors/getItemsList/getItemsList';
import Button from '../../components/Button/Button';
import RecommendModal from '../../components/RecommendModal/RecommendModal';
import { Link } from 'react-router-dom';
import { CheckActions } from '../../redux/CheckSlice/slice/checkSlice';
import RecItem from '../../components/RecItem/RecItem';

const CheckPage = () => {
  const [isOpen, setIsOpen] = useState(false);
  const itemsList = useSelector(getCheckItemsList);
  const dispatch = useDispatch();

  const total = useMemo(
    () => itemsList.reduce((currentTotal, item) => {
      return currentTotal + item.price * item.quantity
    }, 0),
  [itemsList])

  const openHandler = () => {
    setIsOpen(true);
  }

  const closeHandler = () => {
    setIsOpen(false);
  }

  useEffect(() => {
    dispatch(CheckActions.clearData());
  }, [])

  return (
    <>
      <div className={cls.wrapper}>
        <div className={cls.container}>
          <Link to={'/'} className={cls.link}>Вернуться</Link>
          <CheckHeader />
          <RecItem />
          <CheckList />
          <div className={cls.bottom}>
            <div>
              Итого {total ? `${total} руб.` : ''}
            </div>
            <div>
              <Button
                onClick={openHandler}
                disabled={total ? false : true}
              >
                К оплате
              </Button>
            </div>
          </div>
        </div>
      </div>
      <RecommendModal
        isOpen={isOpen}
        onClose={closeHandler}
      />
    </>
  )
}

export default CheckPage;