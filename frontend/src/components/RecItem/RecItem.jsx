import React from 'react';
import cls from './RecItem.module.scss';
import { useSelector } from 'react-redux';
import { getCheckRecItem } from '../../redux/CheckSlice/selectors/getRecItem/getRecItem';
import Loader from '../Loader/Loader';

const RecItem = () => {
  const recItem = useSelector(getCheckRecItem);
  const recLoading = useSelector(state => state.check.recLoading);
  const recError = useSelector(state => state.check.recError || '');

  if (recLoading) {
    return (
      <div className={cls.wrapper}>
        <Loader></Loader>
      </div>
    )  
  }

  if (recError) {
    return (
      <div className={cls.wrapper}>
        <div className={cls.plug}>{recError}</div>
      </div>
    )
  }

  return (
    <div className={cls.wrapper}>
      {recItem?.item_id ? 
        (<div className={cls.item}>
          <div>{recItem.item_id}</div>
          <div>{recItem.name}</div>
          <div className={cls.price}>{parseFloat(recItem.price).toFixed(2)} руб.</div>
        </div>)
      : null}
    </div>
  )
}

export default RecItem;