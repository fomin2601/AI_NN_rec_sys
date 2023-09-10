import React, { useEffect, useMemo, useState } from 'react';
import Modal from '../Modal/Modal';
import { useDispatch, useSelector } from 'react-redux';
import { getCheckRecItem } from '../../redux/CheckSlice/selectors/getRecItem/getRecItem';
import cls from './RecommendModal.module.scss';
import { getCheckItemsList } from '../../redux/CheckSlice/selectors/getItemsList/getItemsList';
import Button from '../Button/Button';
import { CheckActions } from '../../redux/CheckSlice/slice/checkSlice';
import { useNavigate } from 'react-router-dom';

const RecommendModal = (props) => {
  const {
    isOpen,
    onClose
  } = props;

  const dispatch = useDispatch();

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => {
        dispatch(CheckActions.clearData());
        onClose();
      }, 2000)
    }
  }, [isOpen])


  return (
    <Modal
      isOpen={isOpen}
    >
      <div>Оплата успешно</div>
    </Modal>
  )
}

export default RecommendModal;