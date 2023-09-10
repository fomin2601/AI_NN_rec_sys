import React, { useState } from 'react';
import cls from './MainPage.module.scss';
import Button from '../../components/Button/Button';
import { Link } from 'react-router-dom';
import Loader from '../../components/Loader/Loader';
import FileInput from '../../components/FileInput/FileInput';
import axios from 'axios';
import { useDispatch } from 'react-redux';
import { CheckActions } from '../../redux/CheckSlice/slice/checkSlice';
import { classNames } from '../../helpers/ClassNames/classNames';

const MainPage = () => {
  const [clickable, setClickable] = useState(true);
  const [isLoadingDataset, setIsLoadingDataset] = useState(false);
  const [isLoadingChecks, setIsLoadingChecks] = useState(false);
  const [checks, setChecks] = useState([]);
  const dispatch = useDispatch();

  const changeDatasetHandler = async (event) => {
    setClickable(false);
    setIsLoadingDataset(true);
    try {
      const files = new FormData();
      files.append('dataset', event.target.files[0]);
      const url = `http://127.0.0.1:5000/train_model`;
      const response = await fetch(url, {
        method: "POST",
        cache: "no-cache",
        credentials: "same-origin",
        mode: 'cors',
        body: files
      });
      const data = await response.json();
      // console.log('data', data);
      dispatch(CheckActions.setData(data));
    } catch (e) {
      console.log(e);
    } finally {
      setClickable(true);
      setIsLoadingDataset(false); 
    }
  }

  const checksChangeHandler = async (event) => {
    setClickable(false);
    setIsLoadingChecks(true);
    try {
      const file = new FormData();
      file.append(`dataset`, event.target.files[0]);
      const url = `http://127.0.0.1:5000/get_test_predictions`;
      const response = await fetch(url, {
        method: "POST",
        cache: "no-cache",
        credentials: "same-origin",
        mode: 'cors',
        body: file
      });
      const data = await response.json();
      // console.log('data type', typeof data);
      // console.log('data array check', Array.isArray(data));
      if (data.length > 0) {
        setChecks(data);
      }
      // console.log(data);
    } catch (e) {
      console.log(e);
    } finally {
      setClickable(true);
      setIsLoadingChecks(false);
    }
  }

  // console.log('checks', Array.isArray(checks));

  return (
    <div className={cls.wrapper}>
      <div className={cls.container}>
        <div className={cls.btns}>
          <div>
            <FileInput
              title={'Обучить модель'}
              accept={'.tsv'}
              name={'dataset'}
              changeHandler={changeDatasetHandler}
              disabled={!clickable}
            />
            {isLoadingDataset ? <Loader></Loader> : null}
          </div>
          {/* <Button disabled={true}>Датасет</Button> */}
          <div>
            <Link to={'/check'} className={cls.link}>
              <Button disabled={!clickable}>Тест</Button>
            </Link>
          </div>
          <div>
            <FileInput
              title={'Обработать чеки'}
              accept={'.tsv'}
              name={'checks'}
              changeHandler={checksChangeHandler}
              disabled={!clickable}
            />
            {isLoadingChecks ? <Loader></Loader> : null}
          </div>
        </div>
        {
          checks.length > 0 ?
            <div className={cls.checks}>
              <div className={cls.check}>
                <div className={cls.title}>{'receipt_id'}</div>
                <div className={cls.title}>{'item_id'}</div>
              </div>
              <div className={cls.checkList}>
                {checks?.map((check, index) => 
                  (<div key={index} className={cls.check}>
                    <div className={cls.receipt}>{check.receipt_id}</div>
                    <div>{check.item_id}</div>
                  </div>)
                )}
              </div>
            </div>
          : null
        }
      </div>
    </div>
  )
}

export default MainPage;