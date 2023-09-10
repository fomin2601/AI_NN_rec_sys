import React from 'react';
import cls from './CheckHeader.module.scss';
import { useDispatch, useSelector } from 'react-redux';
import { CheckActions } from '../../redux/CheckSlice/slice/checkSlice';
import { getCheckInputItem } from '../../redux/CheckSlice/selectors/getInputItem/getInputItem';
import Button from '../Button/Button';

const CheckHeader = () => {
  const inputId = useSelector(state => state.check.inputId);
  const inputItem = useSelector(getCheckInputItem);
  const dispatch = useDispatch();

  const changeHandler = (event) => {
    dispatch(CheckActions.changeInputId(event.target.value.trim()));
  }

  const clickHandler = () => {
    dispatch(CheckActions.addItemToList({id: inputId, inputItem}))
  }

  return (
    <div className={cls.wrapper}>
      <input 
        name="itemId"
        placeholder={"Введите item_id"}
        className={cls.input}
        onChange={changeHandler}
        value={inputId}
      />
      {
        inputItem?.name ?
          <div className={cls.item}>
           {inputItem.name} {parseFloat(inputItem.price).toFixed(2)} руб.
          </div>
        : <div></div>
      }
      <Button
        disabled={inputItem?.name ? false : true}
        onClick={clickHandler}
      >
        Добавить
      </Button>
    </div>
  )
}

export default CheckHeader;